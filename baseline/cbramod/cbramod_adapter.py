import logging
from typing import List, Dict, Any, Union

import torch
import scipy.signal
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory


logger = logging.getLogger('baseline')


class CBraModDatasetAdapter(AbstractDatasetAdapter):
    def _setup_adapter(self):
        """Initialize CBraMod-specific adapter configurations."""
        self.model_name = 'cbramod'
        self.scale = 0.01
        self.freq = 256
        self.patch_size = 200

        self._log_adapter_info()

    def _log_adapter_info(self):
        """Log adapter initialization information."""
        logger.info(f"{self.model_name}: Dataset Adapter analysis complete:")
        logger.info(f"  - Total samples: {len(self.dataset)}")

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, str, List[str], int]]:
        x: torch.Tensor = sample['data']  # Shape: (n_channels, n_timepoints)

        n_patch = x.shape[1] // self.freq
        data = scipy.signal.resample(x, n_patch * self.patch_size, axis=1)
        data = data * self.scale

        result = {
            'data': data,
            'montage': sample['montage'],
            'chs': sample['chs'],
            'task': sample['task'],
            'label': sample['label'],
        }

        return result

    def get_supported_channels(self) -> List[str]:
        pass


class CBraModDataLoaderFactory(AbstractDataLoaderFactory):
    """CBraMod DataLoader factory that inherits from AbstractDataLoaderFactory."""

    def create_adapter(
            self,
            dataset: HFDataset,
            dataset_names: list[str],
            dataset_configs: list[str]
    ) -> AbstractDatasetAdapter:
        return CBraModDatasetAdapter(dataset, dataset_names, dataset_configs)