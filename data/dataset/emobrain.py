import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union, Any

import datasets
import pytz
from numpy import ndarray
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class EmobrainConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "This project aimed to develop multimodal emotion detection techniques using brain signals (EEG, fNIRS), "
        "Two main datasets were constructed in the paper. The Video - fNIRS database contains data from "
        "16 subjects (6 female, 10 male, average age 25). Video data is stored frame - by - frame with filenames "
        "indicating relevant information. For the EEG + fNIRS recordings, data from 5 male subjects is divided into sessions. "
        "Each session has files for EEG, fNIRS, and self - assessments. ")
    citation: Optional[str] = """\
    @article{article,
    author = {Savran, Arman and Çiftçi, Koray and Chanel, Guillaume and Mota, Javier and Viet, Luong and Sankur, Bulent and Akarun, Lale and Caplier, Alice and Rombaut, Michèle},
    year = {2006},
    month = {01},
    pages = {},
    title = {Emotion Detection in the Loop from Brain Signals and Facial Images}
    }
    """

    filter_notch: float = 50.0

    dataset_name: Optional[str] = 'emobrain'
    task_type: DatasetTaskType = DatasetTaskType.EMOTION
    file_ext: str = 'bdf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        'biosemi64': [
                                        'AF3', 'AF4',
                                'F3', 'F1', 'Fz', 'F2', 'F4',
                'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
                    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
              'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
                              'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                      'O1', 'Oz', 'O2',
                                            'Iz',
        ]
    })

    valid_ratio: float = 0.2
    test_ratio: float = 0.0
    wnd_div_sec: int = 10
    suffix_path: str = os.path.join('EmoBrain', 'eNTERFACE06_EMOBRAIN')
    scan_sub_dir: str = os.path.join('Data', 'EEG')

    category: list[str] = field(default_factory=lambda: [])


class EmobrainBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = EmobrainConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, _, session, _, _, date = file_name.split('_')
        subject = subject[-1]
        session = session[-1]
        data_obj = datetime.strptime(date, "%d%m%Y")
        time_zone = pytz.timezone('Asia/Istanbul')
        data_obj = time_zone.localize(data_obj)
        return {
            'subject': int(subject),
            'session': int(session),
            'date': data_obj.strftime("%d%m%Y"),
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': 'biosemi64',
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

        chs_std = [ch.upper() for ch in self.config.montage[montage]]
        self._std_chs_cache[montage] = chs_std
        return chs_std


if __name__ == "__main__":
    builder = EmobrainBuilder()
    builder.preproc()
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
