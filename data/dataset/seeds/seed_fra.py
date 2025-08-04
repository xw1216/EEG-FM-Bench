import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import numpy as np
import pandas as pd
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class SeedFraConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "The SEED FRA dataset focuses on emotion recognition. "
        "It uses 21 French - language film clips (positive, neutral, negative) as stimuli. "
        "Eight French subjects (5 males, 3 females) participated. "
        "EEG signals and eye movements were collected. "
        "Movie clips are labeled for emotions.")
    citation: Optional[str] = """\
    @article{liu2022identifying, 
    title={Identifying similarities and differences in emotion recognition with EEG and eye movements among Chinese, German, and French People}, 
    author={Liu, Wei and Zheng, Wei-Long and Li, Ziyi and Wu, Si-Yuan and Gan, Lu and Lu, Bao-Liang}, 
    journal={Journal of Neural Engineering}, 
    volume={19}, 
    number={2}, 
    pages={026012}, 
    year={2022}, 
    publisher={IOP Publishing}}
    """

    filter_notch: float = 50.0

    dataset_name: Optional[str] = 'seed_fra'
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

    valid_ratio: float = 0.125
    test_ratio: float = 0.125
    wnd_div_sec: int = 10
    suffix_path: str = os.path.join('SEED', 'SEED_FRA')
    scan_sub_dir: str = os.path.join('French', '01-EEG-raw', 'resampled')

    category: list[str] = field(default_factory=lambda: [
        'negative', 'neutral', 'positive'
    ])


class SeedFraBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = SeedFraConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        # meta_path = os.path.join(self.config.raw_path, 'meta')
        self.sub_meta = pd.DataFrame({
            'subject': [i for i in range(1, 9)],
            'sex': ['F', 'M', 'F', 'M', 'M', 'F', 'M', 'M'],
            'age': [22, 21, 20, 22, 29, 21, 23, 22, ]
        })

        self.time_meta = np.array([
            [21.0, 213.0, 489.0, 687.0, 982.0, 1103.0, 1252.0, 1572.0, 1738.0, 1887.0, 2174.0, 2327.0, 2492.0,
             2619.0, 2783.0, 2931.0, 3029.0, 3216.0, 3369.0, 3515.0, 3622.0, ],
            [193.0, 468.0, 667.0, 962.0, 1082.0, 1232.0, 1552.0, 1718.0, 1867.0, 2154.0, 2307.0, 2471.0, 2599.0,
             2763.0, 2910.0, 3008.0, 3195.0, 3349.0, 3495.0, 3602.0, 3797.0, ]
        ])

        self.label_meta = np.array([
            [1, -1, 0, -1, 1, 0, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0, 1, ],
        ])
        self.label_meta += 1

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, session = file_name.split('_')
        return {
            'subject': int(subject),
            'session': int(session),
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        sex, age = self.sub_meta.loc[info['subject'] - 1, ['sex', 'age']]
        sex = 1 if sex == 'M' else 2
        with self._read_raw_data(file_path, preload=False, verbose=False) as data:
            time = data.times[-1] + data.times[1]

        info.update({
            'montage': '10_10',
            'time': time,
            'sex': sex,
            'age': age,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        labels = self.label_meta[0]
        times = self.time_meta

        annotations = []
        for i in range(len(labels)):
            annotations.append((
                self.config.category[labels[i]],
                round(times[0][i].item() * 1000),
                round(times[1][i].item() * 1000),
            ))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        return self.config.montage[montage]


if __name__ == "__main__":
    builder = SeedFraBuilder('finetune')
    builder.clean_disk_cache()
    builder.preproc()
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
