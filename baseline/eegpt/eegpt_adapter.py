"""
EEGPT Adapter that inherits from AbstractDatasetAdapter.
"""

import logging
from typing import List

from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory, StandardEEGChannelsMixin


logger = logging.getLogger("baseline")


class EegptDatasetAdapter(AbstractDatasetAdapter, StandardEEGChannelsMixin):
    """EEGPT dataset adapter that inherits from AbstractDatasetAdapter."""
    
    def _setup_adapter(self):
        """Initialize EEGPT-specific adapter configurations."""
        self.model_name = 'eegpt'
        self.scale = 0.001  # convert uV to mV
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        """Return list of channels supported by EEGPT."""
        return self.get_standard_eeg_channels()


class EegptDataLoaderFactory(AbstractDataLoaderFactory):
    """EEGPT DataLoader factory that inherits from AbstractDataLoaderFactory."""
    
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str]
    ) -> EegptDatasetAdapter:
        return EegptDatasetAdapter(dataset, dataset_names, dataset_configs)
