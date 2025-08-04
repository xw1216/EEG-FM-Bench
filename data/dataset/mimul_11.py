import logging
import os
import warnings
from dataclasses import dataclass, field
from math import floor
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class Mimul11Config(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "In this work, we collected data on intuitive upper extremity movements from 25 participants. "
        "To collect high-quality signal data, the experiments were conducted on healthy participants, "
        "who had maintained good physical condition by, e.g., limiting their alcohol intake and getting sufficient "
        "sleep. We focused on various upper extremity motions because they are the most extensible and available "
        "movements among all the body movements. Accordingly, we selected the upper extremities for decoding intuitive "
        "movements and then collected data based on the movement-based multimodal signals. The participants were asked "
        "to perform 11 different movement tasks: arm-reaching along 6 directions, hand-grasping of 3 objects, and "
        "wrist-twisting with 2 different motions. The corresponding 11 classes were designed for each segmented motion "
        "related to the arm, hand, and wrist, rather than for continuous limb movements. Therefore, the users of our "
        "dataset could either conduct respective analyses for individual classes or attempt decoding the complex upper "
        "extremity movements by combining data from different classes. For researchers focused on more advanced and "
        "analytical approaches using multimodal signals, the dataset comprised not only EEG data but also "
        "electromyography (EMG) and electro-oculography (EOG) data. These data were synchronously collected in the "
        "same experimental environment, while ensuring no unintentional interference between them. The data acquired "
        "using a 60-channel EEG, 7-channel EMG, and 4-channel EOG were simultaneously recorded during the experiment. "
        "EEG sensors were placed according to international specifications to collect signals from all the regions of "
        "the scalp.")
    citation: Optional[str] = """\
    @article{10.1093/gigascience/giaa098,
    author = {Jeong, Ji-Hoon and Cho, Jeong-Hyun and Shim, Kyung-Hwan and Kwon, Byoung-Hee and Lee, Byeong-Hoo and Lee, Do-Yeun and Lee, Dae-Hyeok and Lee, Seong-Whan},
    title = {Multimodal signal dataset for 11 intuitive movement tasks from single upper extremity during multiple recording sessions},
    journal = {GigaScience},
    volume = {9},
    number = {10},
    pages = {giaa098},
    year = {2020},
    month = {10},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giaa098},
    url = {https://doi.org/10.1093/gigascience/giaa098},}
    """

    """
    !IMPORTANT Errors in original RawData.tar.gz
    Some .vhdr files do not have correct related filename information.
    You would like to open and fix them manually.
    
    File List:
    session1_sub23_twist_MI.vhdr
    session1_sub23_twist_realMove.vhdr
    session2_sub6_reaching_realMove.vhdr
    session2_sub6_multigrasp_MI.vhdr
    session2_sub6_multigrasp_realMove.vhdr
    session2_sub6_twist_realMove.vhdr
    session2_sub6_twist_MI.vhdr
    session2_sub7_twist_MI.vhdr
    session2_sub9_reaching_MI.vhdr
    session2_sub12_twist_MI.vhdr
    session2_sub12_twist_realMove.vhdr
    session2_sub18_multigrasp_realMove.vhdr
    session3_sub10_multigrasp_MI.vhdr 
    session3_sub13_reaching_MI.vhdr
    session3_sub13_reaching_realMove.vhdr
    session3_sub13_twist_MI.vhdr
    session3_sub13_multigrasp_realMove.vhdr
    session3_sub13_twist_realMove.vhdr
    session3_sub17_reaching_MI.vhdr
    session3_sub17_reaching_realMove.vhdr
    session3_sub21_reaching_MI.vhdr
    """

    filter_notch: float = 50.0
    is_notched: bool = True

    dataset_name: Optional[str] = 'mimul_11'
    task_type: DatasetTaskType = DatasetTaskType.MOTOR_IMAGINARY
    file_ext: str = 'vhdr'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                                    'Fp1', 'Fp2',
                          'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                   'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8',
                'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
                'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                          'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                  'O1', 'Oz', 'O2',
                                        'Iz',
        ]
    })

    valid_ratio: float = 0.12
    test_ratio: float = 0.12
    wnd_div_sec: int = 5
    suffix_path: str = 'Multimodal 11 intuitive movement'
    scan_sub_dir: str = "RawData"

    category: list[str] = field(default_factory=lambda: ['reach', 'grasp', 'twist'])


class Mimul11Builder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = Mimul11Config
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True),
        BUILDER_CONFIG_CLASS(name='finetune-reach', is_finetune=True,
                             category=['forward', 'backward', 'left', 'right', 'up', 'down']),
        BUILDER_CONFIG_CLASS(name='finetune-grasp', is_finetune=True,
                             category=['cylindrical', 'spherical', 'lumbrical']),
        BUILDER_CONFIG_CLASS(name='finetune-twist', is_finetune=True,
                             category=['left', 'right']),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        self.sub_meta = {
            'age': [
                26, 25, 29, 28, 29, 30, 32, 30, 31, 32,
                26, 26, 24, 25, 26, 27, 25, 25, 28, 31,
                30, 30, 28, 27, 25,
            ],
            'sex': [
                'M', 'F', 'M', 'M', 'M', 'F', 'M', 'F', 'F', 'M',
                'F', 'M', 'F', 'M', 'M', 'M', 'M', 'F', 'F', 'F',
                'M', 'M', 'F', 'M', 'M',
            ],
        }

    def _walk_raw_data_files(self):
        logger.info('Walking brainvision eeg data files...')
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        if self.config.name == 'finetune-reach':
            prefix = ['reaching']
        elif self.config.name == 'finetune-grasp':
            prefix = ['multigrasp']
        elif self.config.name == 'finetune-twist':
            prefix = ['twist']
        else:
            prefix = ['reaching', 'multigrasp', 'twist']

        raw_data_files = []
        for root, dirs, files in os.walk(scan_path):
            for file in files:
                if file.endswith(self.config.file_ext) and 'MI' in file:
                    skip_flag = True
                    for p in prefix:
                        if p in file:
                            skip_flag = False
                    if skip_flag:
                        continue

                    file_path = os.path.join(root, file)
                    raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        filename = self._extract_file_name(file_path)
        splits = filename.split('_')[:-1]
        session = int(splits[0][7:])
        subject = int(splits[1][3:])
        group = splits[2]

        if group == 'reaching':
            group = 'reach'
        elif group == 'multigrasp':
            group = 'grasp'

        return {
            'subject': subject,
            'session': session,
            'group': group,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        age = self.sub_meta['age'][info['subject'] - 1]
        sex = self.sub_meta['sex'][info['subject'] - 1]

        montage = '10_20'
        with self._read_raw_data(file_path) as raw:
            time = raw.duration

        info.update({
            'montage': montage,
            'time': time,
            'sex': sex,
            'age': age,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        group = info['group']
        raw: BaseRaw = self._read_raw_data(file_path)
        events = self._find_annotation(raw, group)

        annotations = []
        for event in events:
            name = event[0]
            onset = event[1]
            start = floor((onset - 0.5) * 1000)
            end = floor((onset + 4.5) * 1000)

            annotations.append((name, start, end))

        return annotations

    def _find_annotation(self, raw: BaseRaw, group: str):
        if not raw.annotations:
            raise ValueError(f"No annotations found")

        if group == 'reach':
            trigger_mapping: dict[str, str] = {
                'S 11': 'forward',
                'S 21': 'backward',
                'S 31': 'left',
                'S 41': 'right',
                'S 51': 'up',
                'S 61': 'down',
            }
        elif group == 'grasp':
            trigger_mapping: dict[str, str] = {
                'S 11': 'cylindrical',
                'S 21': 'spherical',
                'S 61': 'lumbrical',
            }
        elif group == 'twist':
            trigger_mapping: dict[str, str] = {
                'S 91': 'left',
                'S101': 'right',
            }
        else:
            raise ValueError(f'No such group {group}')

        prefix = 'Stimulus/'
        events = []
        for signal in trigger_mapping.keys():
            descriptions = raw.annotations.description
            matching_indices = np.where(descriptions == f'{prefix}{signal}')[0]
            onset_times = raw.annotations.onset[matching_indices].tolist()

            if self.config.name == 'finetune':
                for onset in onset_times:
                    events.append((group, onset))
            else:
                for onset in onset_times:
                    events.append((trigger_mapping[signal], onset))

        return events

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.upper() for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_brainvision(file_path, preload=preload, verbose=verbose)
            return raw

if __name__ == "__main__":
    builder = Mimul11Builder("finetune-reach")
    # builder.clean_disk_cache()
    builder.preproc(n_proc=1)
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
