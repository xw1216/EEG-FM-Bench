import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
import pandas as pd
import pytz
from mne.io import BaseRaw
from numpy import ndarray
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class SeedVIIConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "SEED-VII is a multimodal emotion recognition dataset that captures EEG and eye movement data "
        "from 20 subjects (10 males, 10 females) aged 19-26. The experiment used 80 video clips to "
        "induce six basic emotions plus neutral states. Each subject underwent four sessions, "
        "with 20 trials per session. Each trial involved watching a video clip (2-5 minutes) followed "
        "by self-assessment. Subjects rated the emotion induction effectiveness on a 0-1 scale.")
    citation: Optional[str] = """\
    @ARTICLE{10731546, author={Jiang, Wei-Bang and Liu, Xuan-Hao and Zheng, Wei-Long and Lu, Bao-Liang}, 
    journal={IEEE Transactions on Affective Computing}, 
    title={SEED-VII: A Multimodal Dataset of Six Basic Emotions with Continuous Labels for Emotion Recognition}, 
    year={2024}, 
    volume={}, 
    number={}, 
    pages={1-16}, 
    keywords={Electroencephalography;Emotion recognition;Brain modeling;Physiology;Videos;Electrocardiography;Transformers;Recording;Computational modeling;Affective computing;Basic emotions;EEG;eye movements;emotion recognition;continuous label;multimodal dataset}, 
    doi={10.1109/TAFFC.2024.3485057}}
    """

    filter_notch: float = 50.0
    persist_drop_last: bool = False

    dataset_name: Optional[str] = 'seed_vii'
    task_type: DatasetTaskType = DatasetTaskType.EMOTION
    file_ext: str = 'cnt'
    montage: dict[str, list[str]] = field(default_factory=lambda : {
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

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 15
    suffix_path: str = os.path.join('SEED', 'SEED-VII', 'SEED-VII')
    scan_sub_dir: str = "EEG_raw"

    category: list[str] = field(default_factory=lambda:
    ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy', 'Anger', 'Surprise'])

    is_cross_subject: bool = True


class SeedVIIBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = SeedVIIConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()
        self.save_info_path = os.path.join(self.config.raw_path, 'save_info')

    def _load_meta_info(self):
        # meta_path = os.path.join(self.config.raw_path, 'meta')
        self.sub_meta = pd.DataFrame({
            'subject': [i for i in range(1, 21)],
            'sex': ['M', 'M', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'M'],
            'age': [26, 24, 25, 25, 20, 22, 22, 21, 24, 20, 24, 23, 21, 19, 22, 21, 23, 23, 23, 22]
        })

        self.label_meta = np.array([
            [4, 3, 0, 2, 5, 5, 2, 0, 3, 4, 4, 4, 0, 5, 2, 5, 2, 0, 3, 4, ],
            [5, 2, 1, 3, 6, 6, 3, 1, 2, 5, 5, 2, 1, 3, 6, 6, 3, 1, 2, 5, ],
            [4, 6, 0, 1, 5, 5, 1, 0, 6, 4, 4, 6, 0, 1, 5, 5, 1, 0, 6, 4, ],
            [0, 2, 1, 6, 4, 4, 6, 1, 2, 0, 0, 2, 1, 6, 4, 4, 6, 1, 2, 0, ]
        ])

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, date, session = file_name.split('_')
        data_obj = datetime.strptime(date, "%Y%m%d")
        time_zone = pytz.timezone('Asia/Shanghai')
        data_obj = time_zone.localize(data_obj)
        return {
            'subject': int(subject),
            'session': int(session),
            'date': data_obj.strftime("%Y%m%d"),
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        sex, age = self.sub_meta.loc[info['subject'] - 1, ['sex', 'age']]
        sex = 1 if sex == 'M' else 2
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': '10_10',
            'time': time,
            'sex': sex,
            'age': age,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        session = info['session']
        labels = self.label_meta[session - 1]

        file_name = self._extract_file_name(file_path)

        if file_name in ['9_20221111_3', '14_20221015_1']:
            trigger_df = pd.read_csv(
                os.path.join(self.save_info_path, file_name + '_trigger_info.csv'),
                header=None, sep=',',
                names=['event', 'datetime'])
            # trigger_df['timestamp'] = pd.to_datetime(trigger_df['datetime'])
            start_time = pd.to_datetime('2022-11-11 14:01:27') \
                if file_name == '9_20221111_3' else pd.to_datetime('2022-10-15 14:25:34')
            trigger_df['time_diff'] = (pd.to_datetime(trigger_df['datetime']) - start_time).dt.total_seconds()
            # trigger_df['time_diff'] = trigger_df['time_diff'].dt.total_seconds()
            onsets: ndarray = trigger_df['time_diff'].to_numpy()
        else:
            onsets: ndarray = self._read_raw_data(file_path, preload=False, verbose=False).annotations.onset

        onsets = onsets[: 2 * len(labels)]
        annotations = []
        for label, onset in zip(labels, onsets.reshape([-1, 2])):
            annotations.append((
                self.config.category[label],
                round(onset[0].item() * 1000),
                round(onset[1].item() * 1000),
            ))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        return self.config.montage[montage]

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            data = mne.io.read_raw_cnt(file_path, preload=preload, verbose=verbose)
            return data


if __name__ == "__main__":
    builder = SeedVIIBuilder('finetune')
    builder.clean_disk_cache()
    builder.preproc()
    builder.download_and_prepare(num_proc=4)
    dataset = builder.as_dataset()
    print(dataset)
