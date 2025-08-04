import glob
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class SeedConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "The SEED dataset is a multimodal dataset for emotion recognition "
        "research. It contains physiological signal data and corresponding "
        "emotion labels. 15 Chinese movie clips were carefully selected as "
        "stimuli to evoke different emotions.")
    citation: Optional[str] = """\
    @ARTICLE{7104132,
    author={Zheng, Wei-Long and Lu, Bao-Liang},
    journal={IEEE Transactions on Autonomous Mental Development}, 
    title={Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition with Deep Neural Networks}, 
    year={2015},
    volume={7},
    number={3},
    pages={162-175},
    keywords={Electroencephalography;Feature extraction;Brain modeling;Electrodes;Emotion recognition;Entropy;Affective computing;deep belief networks;EEG;emotion recognition},
    doi={10.1109/TAMD.2015.2431497}}
    """

    filter_notch: float = 50.0
    orig_fs: float = 200.0

    dataset_name: Optional[str] = 'seed'
    task_type: DatasetTaskType = DatasetTaskType.EMOTION
    file_ext: str = 'set'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_10': [
                              'FP1','FPZ','FP2',
                              'AF3',    'AF4',
                'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
            'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
                'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
            'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
                'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
                  'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
                               'O1','OZ','O2'
        ]
    })

    valid_ratio: float = 0.067
    test_ratio: float = 0.067
    wnd_div_sec: int = 10
    suffix_path: str = os.path.join('SEED', 'SEED')
    scan_sub_dir: str = os.path.join('SEED_EEG', 'SEED_RAW_EEG', 'resampled')
    category: list[str] = field(default_factory=lambda: ['sad', 'neutral', 'happy'])
    is_cross_subject: bool = True


class SeedBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = SeedConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True, wnd_div_sec=4),
        BUILDER_CONFIG_CLASS(name='finetune_sub_dependent', is_finetune=True, wnd_div_sec=4, is_cross_subject=False),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()
        self.n_trial = self.label_meta.shape[1]

    def _load_meta_info(self):
        # meta_path = os.path.join(self.config.raw_path, 'meta')
        self.sub_meta = pd.DataFrame({
            'subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'sex': ['M','F','F','M','M','M','F','F','M','F','F','M','F','M','F',]
        })
        self.label_meta: ndarray = np.array([
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
        ])

        self.time_meta = np.array([
            [27, 290, 551, 784, 1050, 1262, 1484, 1748, 1993, 2287, 2551, 2812, 3072, 3335, 3599],
            [262, 523, 757, 1022, 1235, 1457, 1721, 1964, 2258, 2524, 2786, 3045, 3307, 3573, 3805]
        ])

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, session = file_name.split('_')
        return {
            'subject': int(subject),
            'session': int(session),
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        sex = self.sub_meta.loc[info['subject'] - 1, 'sex']
        sex = 1 if sex == 'M' else 2

        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': '10_10',
            'sex': sex,
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        annotations = []
        times = self.time_meta
        labels = self.label_meta[0]

        for i in range(len(labels)):
            annotations.append((
                self.config.category[labels[i]],
                times[0][i].item() * 1000,
                times[1][i].item() * 1000,
            ))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        if self.config.is_cross_subject:
            return self._divide_all_split_by_sub(df)
        else:
            return self._divide_by_uniform_label(df)

    @staticmethod
    def _divide_by_uniform_label(df: DataFrame) -> DataFrame:
        def split_row(row):
            labels = json.loads(row['label'])
            label_train, label_valid, label_test = labels[0:9], labels[9:12], labels[12:15]
            new_rows = []
            for set_name, label in zip(['train', 'valid', 'test'], [label_train, label_valid, label_test]):
                new_row = row.copy()
                new_row['split'] = set_name
                new_row['label'] = json.dumps(label)
                new_rows.append(new_row)
            return pd.DataFrame(new_rows)

        df['split'] = 'train'
        processed = df.apply(split_row, axis=1).tolist()
        new_df = pd.concat(processed).reset_index(drop=True)
        return new_df

    def standardize_chs_names(self, montage: str):
        return self.config.montage[montage]

    def _check_data_montage_channel(self, df: DataFrame, n_proc: Optional[int] = None):
        return df

    def _check_data_length(self, df: DataFrame):
        return df

    @staticmethod
    def _orig_ch_names():
        return [
                              'FP1','FPZ','FP2',
                              'AF3',    'AF4',
                'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
            'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
                'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
            'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
                'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
                  'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
                         'CB1','O1','OZ','O2','CB2'
        ]

    def _convert_to_mne(self, data: ndarray ,info):
        ch_names = self._orig_ch_names()
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.config.orig_fs,
            ch_types=['eeg'] * len(ch_names))

        # mne expected unit is Volts, so turn uV to V
        raw = mne.io.RawArray(data / 1e6, info, verbose=False)
        return raw

if __name__ == "__main__":
    builder = SeedBuilder('finetune_sub_dependent')
    builder.clean_disk_cache()
    builder.preproc()
    builder.download_and_prepare(num_proc=8)
    dataset = builder.as_dataset()
    print(dataset)


