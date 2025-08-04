import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
from mne.io import BaseRaw
from numpy import ndarray
from pandas import DataFrame
from scipy.io import loadmat

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class SpisRestingStateConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "These 10 datasets were recorded prior to a 105-minute session of Sustained Attention to Response Task with "
        "fixed-sequence and varying ISIs. Each dataset contains 2.5 minutes of eyes-open (EO) and 2.5 minutes of "
        "eyes-closed (EC) resting-state EEG. Monopolar EEG activity was collected at 2,048 Hz via 64 Ag/AgCl "
        "active electrodes mounted according to the 10-10 International Electrode Placement System. Experiments were "
        "conducted in the early afternoon to induce drowsiness in the already idled brain networks.")
    citation: Optional[str] = """\
    @ARTICLE{9034192,
    author={Torkamani-Azar, Mastaneh and Kanik, Sumeyra Demir and Aydin, Serap and Cetin, Mujdat},
    journal={IEEE Journal of Biomedical and Health Informatics}, 
    title={Prediction of Reaction Time and Vigilance Variability From Spatio-Spectral Features of Resting-State EEG in a Long Sustained Attention Task}, 
    year={2020},
    volume={24},
    number={9},
    pages={2550-2558},
    keywords={Task analysis;Electroencephalography;Informatics;Time factors;Monitoring;Brain modeling;Electrodes;Brain-computer interface;resting-state analysis;electroencephalography;neural networks;multivariate regression;human performance;sustained attention;vigilance;default mode network},
    doi={10.1109/JBHI.2020.2980056}}
    """

    filter_notch: float = 50.0
    orig_fs: float = 256.0

    dataset_name: Optional[str] = 'spis_resting_state'
    task_type: DatasetTaskType = DatasetTaskType.RESTING
    file_ext: str = 'mat'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        'biosemi64': [
                                    'FP1', 'FPZ', 'FP2',
                             'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
            'FT7', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8',
                      'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
            'TP7', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8',
                      'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
             'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
                             'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
                                     'O1', 'O2', 'OZ',
                                           'IZ'
        ]
    })

    valid_ratio: float = 0.1
    test_ratio: float = 0.0
    wnd_div_sec: int = 10
    suffix_path: str = os.path.join('SPIS Resting State Dataset', 'SPIS-Resting-State-Dataset-master')
    scan_sub_dir: str = "Pre-SART EEG"

    category: list[str] = field(default_factory=lambda: [
        'eye_closed', 'eye_open',
    ])


class SpisRestingStateBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = SpisRestingStateConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self.header_column = [
            'FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7',
            'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ',
            'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
            'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',
            'P10', 'PO8', 'PO4', 'O2']

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        data: ndarray = loadmat(file_path)['dataRest']
        data = data[:64, :] * 1e-3
        data = data - data.mean(axis=1, keepdims=True)
        raw = self._convert_to_mne(data, None)
        return raw

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, _, session = file_name.split('_')
        subject = int(subject[1:])
        session = 0 if session == 'EC' else 1
        return {
            'subject': subject,
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        data = loadmat(file_path)['dataRest']
        time = data.shape[1] / self.config.orig_fs
        info.update({
            'montage': 'biosemi64',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        session = info['session']
        return [(self.config.category[session], 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        return self.config.montage[montage]

    def _convert_to_mne(self, data: ndarray, info):
        info = mne.create_info(
            ch_names=self.header_column,
            sfreq=self.config.orig_fs,
            ch_types=['eeg'] * len(self.header_column)
        )
        raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
        return raw


if __name__ == "__main__":
    builder = SpisRestingStateBuilder()
    builder.preproc()
    builder.download_and_prepare(num_proc=4)
    dataset = builder.as_dataset()
    print(dataset)
