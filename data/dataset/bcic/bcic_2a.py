import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne.io
import numpy as np
import pandas as pd
import s3fs
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class BCIC2AConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "This data set consists of EEG data from 9 subjects. The cue-based BCI paradigm consisted of "
        "four different motor imagery tasks, namely the imagination of movement of the "
        "left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4). "
        "Two sessions on different days were recorded for each subject. Each session is comprised "
        "of 6 runs separated by short breaks. One run consists of 48 trials (12 for each of the "
        "four possible classes), yielding a total of 288 trials per session.")
    citation: Optional[str] = "https://www.bbci.de/competition/iv/desc_2a.pdf"

    filter_low: float = 0.5
    filter_high: float = 45.0
    filter_notch: float = 50.0
    is_notched: bool = False

    dataset_name: Optional[str] = 'bcic_2a'
    task_type: DatasetTaskType = DatasetTaskType.MOTOR_IMAGINARY

    # !IMPORTANT, in mne 1.9.0 and numpy 2.1.3, gdf reading can result in uint8 out of bound
    # Please refer to https://github.com/mne-tools/mne-python/issues/13111 in mne\io\edf\edf.py at line 1455, then change mne package source code

    # And runs are seperated by multiple NaN in signal data
    file_ext: str = 'set'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                              'Fz',
                  'E2', 'E3', 'E4', 'E5', 'E6',
            'E7', 'C3', 'E9', 'Cz', 'E11', 'C4', 'E13',
                'E14', 'E15', 'E16', 'E17', 'E18',
                       'E19', 'Pz', 'E21',
                              'E22'
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 4
    suffix_path: str = os.path.join('BCI Competition IV', '2a')
    scan_sub_dir: str = "set"

    category: list[str] = field(default_factory=lambda: [
        'left', 'right', 'foot', 'tongue'
    ])


class BCIC2ABuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = BCIC2AConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain', **kwargs):
        super().__init__(config_name, **kwargs)

    def _walk_raw_data_files(self):
        # noinspection PyTypeChecker
        scan_path: str = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []
        for root, dirs, files in os.walk(scan_path):
            for file in files:
                if file.endswith(self.config.file_ext):
                    # if self.config.is_finetune and 'E' in file:
                    #     continue
                    file_path = os.path.join(root, file)
                    raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject = int(file_name[1:3])
        session_type = file_name[3]
        session = int(file_name[-1])
        session = session if session_type == 'T' else session + 10

        return {
            'subject': subject,
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': '10_20',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            anno = raw.annotations
            onset_list = anno.onset
            desc_list = anno.description

        assert len(onset_list) == len(desc_list)

        annotations = []
        for onset, desc in zip(onset_list, desc_list):
            if self.config.is_finetune:
                label = desc
            else:
                label = 'default'

            annotations.append((
                label,
                round(onset * 1000),
                round((onset + self.config.wnd_div_sec) * 1000)
            ))

        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:

        if self.config.is_finetune:
            df.loc[df['subject'].isin(np.array([1, 2, 3, 4, 5])), 'split'] = 'train'
            df.loc[df['subject'].isin(np.array([6, 7])), 'split'] = 'valid'
            df.loc[df['subject'].isin(np.array([8, 9])), 'split'] = 'test'
        else:
            df.loc[df['subject'].isin(np.array([1, 2, 3, 4, 5, 6, 7])), 'split'] = 'train'
            df.loc[df['subject'].isin(np.array([8, 9])), 'split'] = 'valid'

        return df

    def standardize_chs_names(self, montage: str):
        if montage == '10_20':
            return [
                                'FZ',
                  'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
              'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                  'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
                          'P1', 'PZ', 'P2',
                                'POZ',
            ]
        else:
            raise ValueError('No such montage in bcic_2a dataset')

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_eeglab(file_path, preload=preload, verbose=verbose)
            return raw


if __name__ == "__main__":
    builder = BCIC2ABuilder('finetune')
    # builder.clean_disk_cache()
    builder.preproc(n_proc=2)
    builder.download_and_prepare(num_proc=2)
    dataset = builder.as_dataset()
    print(dataset)

