import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import pandas as pd
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class TargetVersusNonConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("0.0.0")
    description: Optional[str] = (
        "This dataset contains electroencephalographic (EEG) recordings of 50 subjects playing to a visual "
        "P300 Brain-Computer Interface (BCI) videogame named Brain Invaders. The interface uses the oddball "
        "paradigm on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed pseudo-randomly to "
        "elicit the P300 response. EEG data were recorded using 32 active wet electrodes with three conditions: "
        "flash duration 50ms, 80ms or 110ms. The experiment took place at GIPSA-lab, Grenoble, France, in 2015. "
        "A full description of the experiment is available at https://hal.archives-ouvertes.fr/hal-02172347. "
        "Python code for manipulating the data is available at https://github.com/plcrodrigues/py.BI.EEG.2015a-GIPSA. "
        "The ID of this dataset is bi2015a.")
    citation: Optional[str] = """\
    @techreport{korczowski:hal-02172347,
    TITLE = {{Brain Invaders calibration-less P300-based BCI with modulation of flash duration Dataset (bi2015a)}},
    AUTHOR = {Korczowski, Louis and Cederhout, Martine and Andreev, Anton and Cattan, Gr{\'e}goire and Coelho Rodrigues, Pedro Luiz and Gautheret, Violette and Congedo, Marco},
    URL = {https://hal.science/hal-02172347},
    TYPE = {Research Report},
    INSTITUTION = {{GIPSA-lab}},
    YEAR = {2019},
    MONTH = Jul,
    DOI = {10.5281/zenodo.3266930},
    KEYWORDS = {Electroencephalography (EEG) ; Brain-Computer Interface ; Experiment ; Electroencephalographie (EEG) ; P300 ; Interface Cerveau-Machine (ICM) ; Interface Cerveau-Ordinateur (ICO) ; Exp{\'e}rimentation},
    PDF = {https://hal.science/hal-02172347v1/file/bi2015a_report_publication_v4.pdf},
    HAL_ID = {hal-02172347},
    HAL_VERSION = {v1},
    }
    """

    filter_notch: float = 50.0
    orig_fs: float = 512.0

    dataset_name: Optional[str] = 'target_versus_non'
    task_type: DatasetTaskType = DatasetTaskType.ERP
    file_ext: str = 'csv'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_10': [
                        'FP1', 'FP2',
                            'AFZ',
                    'F7','F3', 'F4', 'F8',
                'FC5', 'FC1', 'FC2', 'FC6',
                'T7', 'C3', 'CZ', 'C4', 'T8',
                'CP5', 'CP1', 'CP2', 'CP6',
                'P7', 'P3', 'PZ', 'P4', 'P8',
                'PO9', 'PO7', 'PO8', 'PO10',
                      'O1', 'OZ', 'O2',
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.0
    wnd_div_sec: int = 15
    suffix_path: str = os.path.join('Target Versus Non-Target', 'Target non Target bi2015a')
    scan_sub_dir: str = "data"
    category: list[str] = field(default_factory=lambda: [])


class TargetVersusNonBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = TargetVersusNonConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self.header_column = pd.read_csv(os.path.join(os.path.dirname(self.config.raw_path), 'Header.csv')).columns.tolist()

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        df = pd.read_csv(file_path, header=None, names=self.header_column)
        raw = self._convert_to_mne(df, None)
        return raw

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        _, subject, event, order = file_name.split('_')
        if event == 'session':
            session = int(order)
        else:
            session = 4
            if event == 'fixing':
                session += 2
            if order == 'after':
                session += 1
        return {
            'subject': int(subject),
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        data = pd.read_csv(file_path, header=None, names=self.header_column)
        time = len(data) / self.config.orig_fs

        info.update({
            'montage': '10_10',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        return [('default', 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs: list[str] = self.config.montage[montage]
        chs_std = [ch.upper() for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _convert_to_mne(self, df: DataFrame, info) -> mne.io.RawArray:
        ch_names = self.header_column[1:-2]
        std_ch_names = self.standardize_chs_names('10_10')
        signal = df.loc[:, ch_names].to_numpy().transpose(1, 0) / 1e6

        info = mne.create_info(
            ch_names=std_ch_names,
            sfreq=self.config.orig_fs,
            ch_types=['eeg'] * len(ch_names)
        )

        raw = mne.io.RawArray(signal, info, verbose=False)
        return raw

if __name__ == "__main__":
    builder = TargetVersusNonBuilder()
    builder.preproc()
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)

