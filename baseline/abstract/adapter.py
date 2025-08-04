"""
Abstract adapter base class for baseline models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
import datasets

from common.distributed.loader import DistributedGroupBatchSampler
from data.processor.wrapper import load_concat_eeg_datasets, get_dataset_montage


logger = logging.getLogger('baseline')


class AbstractDatasetAdapter(Dataset, ABC):
    """Abstract base adapter for dataset processing."""
    
    def __init__(self, dataset: HFDataset, dataset_names: List[str], dataset_configs: List[str]):
        self.model_name = ''
        self.dataset = dataset
        self.dataset_names = dataset_names
        self.dataset_configs = dataset_configs
        self.montage_mappings = {}

        self.scale = 1.0

        self._setup_adapter()
    
    def _setup_adapter(self):
        """Initialize adapter-specific configurations. Instance property must be pickable"""
        self._build_montage_mappings()
        self._log_adapter_info()
    
    def _build_montage_mappings(self):
        """Build montage mappings with dataset-specific routing. Common implementation."""
        supported_channels = self.get_supported_channels()

        for dataset_name, dataset_config in zip(self.dataset_names, self.dataset_configs):
            montages = get_dataset_montage(dataset_name=dataset_name, config_name=dataset_config)

            for montage_key, channel_names in montages.items():
                # Process montage to create model-specific mapping
                available_indices = []
                available_channels = []

                for ch in supported_channels:
                    if ch in channel_names:
                        idx = supported_channels.index(ch)
                        available_channels.append(ch)
                        available_indices.append(idx)

                if available_channels:  # Only add if we have compatible channels
                    # Use lists instead of tensors for better serialization
                    selector = [False] * len(channel_names)
                    for ch in available_channels:
                        idx = channel_names.index(ch)
                        selector[idx] = True

                    mapping_info = {
                        'idx': available_indices,  # Use list instead of tensor
                        'chs': available_channels,
                        'n_ch': len(channel_names),
                        'sel': selector,  # Use list instead of tensor
                    }

                    self.montage_mappings[montage_key] = mapping_info

                    logger.info(f"Added montage {montage_key} for dataset {dataset_name}: {len(available_channels)} channels")
                else:
                    logger.warning(f"No compatible channels found for montage {montage_key} in dataset {dataset_name}")


    def _log_adapter_info(self):
        """Log adapter initialization information."""
        logger.info(f"{self.model_name}: Dataset Adapter analysis complete:")
        logger.info(f"  - Total samples: {len(self.dataset)}")
        logger.info(f"  - Available montages: {list(self.montage_mappings.keys())}")
        for montage, info in self.montage_mappings.items():
            logger.info(f"  - {montage}: {len(info['chs'])} channels")


    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, str, List[str], int]]:
        """Process a single sample according to model requirements."""
        # Common processing logic
        data: torch.Tensor = sample['data']  # Shape: (n_channels, n_timepoints)
        montage = sample['montage']
        task = sample['task']

        # Map channels using appropriate montage mapping
        if montage not in self.montage_mappings:
            raise ValueError(f"Montage {montage} not found in mappings")
            
        montage_info = self.montage_mappings[montage]
        
        # Convert data to tensor and ensure it's contiguous
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        else:
            data = data.float()
            
        # Apply channel selection using boolean indexing
        selector = torch.tensor(montage_info['sel'], dtype=torch.bool)
        data_mapped = data[selector, :] * self.scale
        
        # Apply model-specific data processing
        data_processed = self._apply_model_specific_processing(data_mapped, montage_info)
        
        # Get selected channel names
        chs = sample['chs'][montage_info['sel']]
        
        # Convert indices to tensor
        chans_id = torch.tensor(montage_info['idx'], dtype=torch.long)
        
        # Build result dictionary
        result = {
            'data': data_processed,
            'montage': montage,
            'chs': chs,
            'chans_id': chans_id,
            'task': task,
            'label': sample['label'],
        }
        
        return result

    def _apply_model_specific_processing(self, data: torch.Tensor, montage_info: Dict[str, Any]) -> torch.Tensor:
        """Apply model-specific data processing. Override in subclasses for custom processing."""
        return data
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, Union[torch.Tensor, str, List[str], int]]:
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        
        sample = self.dataset[idx]
        return self._process_sample(sample)
    
    @abstractmethod
    def get_supported_channels(self) -> List[str]:
        """Return list of channels supported by this model."""
        pass
    
    def get_montage_info(self, montage: str) -> Dict[str, Any]:
        """Get montage information for channel mapping."""
        if montage in self.montage_mappings:
            return self.montage_mappings[montage].copy()
        else:
            raise ValueError(f"Montage {montage} not found")


class AbstractDataLoaderFactory(ABC):
    """Abstract factory for creating data loaders."""
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4, seed: int = 42):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
    
    @abstractmethod
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str]
    ) -> AbstractDatasetAdapter:
        """Create dataset adapter instance."""
        pass
    
    def loading_dataset(
        self,
        datasets_config: Dict[str, str],
        split: datasets.NamedSplit,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        """Create data loader for training/evaluation.
        
        This is a concrete implementation that can be used by most models.
        Override this method for model-specific dataloader creation.
        """
        
        dataset_names = list(datasets_config.keys())
        config_names = list(datasets_config.values())
        
        # Load combined dataset
        combined_dataset, _ = load_concat_eeg_datasets(
            dataset_names=dataset_names,
            builder_configs=config_names,
            split=split,
            cast_label=True
        )

        # Create adapter
        adapter = self.create_adapter(
            dataset=combined_dataset,
            dataset_names=dataset_names,
            dataset_configs=config_names
        )

        sampler = DistributedGroupBatchSampler(
            dataset=combined_dataset,
            batch_size=self.batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=self.seed
        )

        dataloader_kwargs = {
            'batch_sampler': sampler,
            'num_workers': self.num_workers,
            'persistent_workers': self.num_workers > 0,
            'prefetch_factor': 2 if self.num_workers > 0 else None,
        }

        if self.num_workers > 0:
            # noinspection PyTypeChecker
            dataloader_kwargs['multiprocessing_context'] = 'spawn'

        dataloader = torch.utils.data.DataLoader(adapter, **dataloader_kwargs)

        return dataloader, sampler

    def create_dataloader(
        self,
        datasets_config: Dict[str, str],
        mixed: bool,
        num_replicas: int,
        rank: int,
        split: datasets.NamedSplit,
    ) -> tuple[Union[list[DataLoader], DataLoader], Union[list[DistributedGroupBatchSampler], DistributedGroupBatchSampler]]:
        if mixed:
            return self.loading_dataset(
                datasets_config=datasets_config,
                split=split,
                num_replicas=num_replicas,
                rank=rank,
            )
        else:
            dataloaders, samplers = [], []
            for dataset_name, config_name in datasets_config.items():
                loader, sampler = self.loading_dataset(
                    datasets_config={dataset_name: config_name},
                    split=split,
                    num_replicas=num_replicas,
                    rank=rank
                )
                dataloaders.append(loader)
                samplers.append(sampler)
        return dataloaders, samplers


class StandardEEGChannelsMixin:
    """Mixin that provides standard EEG channel list used by most models."""
    
    @staticmethod
    def get_standard_eeg_channels() -> List[str]:
        """Return standard EEG channel list used by most baseline models."""
        return [
            'FP1', 'FPZ', 'FP2',
            'AF7', 'AF3', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
            'O1', 'OZ', 'O2',
        ]
