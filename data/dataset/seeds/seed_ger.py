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
class SeedGerConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "The SEED - GER dataset is for emotion recognition research. "
        "It uses 20 film clips (positive, neutral, negative) as stimuli. "
        "Eight German subjects (7 males, 1 female) participated. "
        "EEG signals and eye - tracking data were collected.")
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

    dataset_name: Optional[str] = 'seed_ger'
    task_type: DatasetTaskType = DatasetTaskType.EMOTION
    # Ubuntu have unknown issue in opening large cnt files using mne
    # So transform to set files using matlab script
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
    suffix_path: str = os.path.join('SEED', 'SEED_GER')
    scan_sub_dir: str = os.path.join('German', '01-EEG-raw', 'resampled')

    category: list[str] = field(default_factory=lambda: [
        'negative', 'neutral', 'positive'
    ])


class SeedGerBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = SeedGerConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        self.sub_meta = pd.DataFrame({
            'subject': [i for i in range(1, 9)],
            'sex': ['M', 'M', 'M', 'M', 'M', 'M', 'F', 'M',],
            'age': [20, 22, 26, 23, 21, 21, 24, 21,]
        })

        self.time_meta = np.array([
            [5.0, 411.0, 861.0, 1114.0, 1287.0, 1454.0, 1620.0, 1878.0, 2135.0, 2310.0, 2502.0, 2709.0, 3028.0, 3162.0, 3290.0, 3656.0, 3823.0, 4366.0, ],
            [136.0, 831.0, 1084.0, 1257.0, 1423.0, 1589.0, 1848.0, 2105.0, 2280.0, 2472.0, 2677.0, 2998.0, 3131.0, 3259.0, 3626.0, 3792.0, 4079.0, 4538.0, ]
        ])

        self.label_meta = np.array([
            [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, 0, -1, -1, 0, 1, 1, ]
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

        labels = self.label_meta[0]
        times = self.time_meta[0]

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
    builder = SeedGerBuilder('finetune')
    builder.preproc()
    builder.download_and_prepare(num_proc=8)
    dataset = builder.as_dataset()
    print(dataset)
