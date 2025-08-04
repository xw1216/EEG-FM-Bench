"""
LABRAM Adapter that inherits from AbstractDatasetAdapter.
"""

import logging
from typing import Dict, List, Union, Any

import torch
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory
from data.processor.wrapper import get_dataset_montage


logger = logging.getLogger("baseline")


class LabramDatasetAdapter(AbstractDatasetAdapter):
    """LABRAM dataset adapter that inherits from AbstractDatasetAdapter."""
    
    def _setup_adapter(self):
        self.model_name = 'labram'
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        """Return list of channels supported by Labram (including wired position in TUEG)."""
        return [
            'FP1', 'FPZ', 'FP2', 
            'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
            'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
            'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
            'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
            'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
            'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
            'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
            'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
            'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
            'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
            'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
            'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", 
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
        ]


class LabramDataLoaderFactory(AbstractDataLoaderFactory):
    """Labram DataLoader factory that inherits from AbstractDataLoaderFactory."""
    
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str]
    ) -> LabramDatasetAdapter:
        return LabramDatasetAdapter(dataset, dataset_names, dataset_configs) 