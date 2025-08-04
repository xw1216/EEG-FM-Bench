import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
import pandas as pd
from mne.io import BaseRaw
from numpy import ndarray
from pandas import DataFrame, Series

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class InriaBciConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "")
    citation: Optional[str] = """\
    @Article{Margaux2012,
    author={Margaux, Perrin
    and Emmanuel, Maby
    and S{\'e}bastien, Daligault
    and Olivier, Bertrand
    and J{\'e}r{\'e}mie, Mattout},
    title={Objective and Subjective Evaluation of Online Error Correction during P300-Based Spelling},
    journal={Advances in Human-Computer Interaction},
    year={2012},
    volume={2012},
    number={1},
    pages={578295},
    note={578295},
    issn={1687-5893},
    doi={10.1155/2012/578295},
    url={https://doi.org/10.1155/2012/578295}
    }
    """

    filter_notch: float = 50.0
    orig_fs: float = 200.0

    dataset_name: Optional[str] = 'inria_bci'
    task_type: DatasetTaskType = DatasetTaskType.ERP
    file_ext: str = 'csv'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_10': [
                                    'Fp1', 'Fp2',
                            'AF7', 'AF3', 'AF4', 'AF8',
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
                'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
                'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                                 'PO7', 'POz', 'PO8',
                                      'O1', 'O2'
        ]
    })

    valid_ratio: float = 0.154
    test_ratio: float = 0.0
    wnd_div_sec: int = 5
    wnd_pre: int = 3
    wnd_post: int = 2

    suffix_path: str = os.path.join('Infra BCI Challenge', 'inria-bci-challenge')
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: [
        'wrong', 'correct'
    ])


class InriaBciBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = InriaBciConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain', valid_ratio=0.154),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True, valid_ratio=0.125, test_ratio=0.125),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self.label_meta: DataFrame = pd.read_csv(os.path.join(self.config.raw_path, 'TrainLabels.csv'))

    def _walk_raw_data_files(self):
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []
        for root, dirs, files in os.walk(scan_path):
            for file in files:
                if file.endswith(self.config.file_ext):
                    if self.config.is_finetune and 'test' in root:
                        continue
                    file_path = os.path.join(root, file)
                    raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        _, subject, session = file_name.split('_')
        subject = int(subject[1:])
        session = int(session[4:])
        return {
            'subject': subject,
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        split = self._extract_middle_path(file_path, -2, -1)[0]

        df = pd.read_csv(file_path, sep=',', header='infer')
        time = df.loc[len(df) - 1, 'Time']
        info.update({
            'montage': '10_10',
            'split': split,
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        df: Series = pd.read_csv(file_path, sep=',', header='infer')
        idx = df.index[df['FeedBackEvent'] == 1].to_numpy()

        start_idx = idx - round(self.config.wnd_pre * self.config.fs)
        end_idx = idx + round(self.config.wnd_post * self.config.fs)
        start = (df.loc[start_idx, 'Time'].values * 1000).astype(np.int64).tolist()
        end = (df.loc[end_idx, 'Time'].values * 1000).astype(np.int64).tolist()

        row_prefix = self._extract_file_name(file_path).split('Data_')[1]
        labels = self.label_meta[self.label_meta['IdFeedBack'].str.contains(row_prefix)]
        labels = np.array(self.config.category)[labels['Prediction']]

        assert len(start) == len(end) == len(labels)
        annotations = list(zip(labels, start, end))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        if self.config.is_finetune:
            df = self._divide_all_split_by_sub(df)
        else:
            df.loc[df['split'] == 'test', 'split'] = 'valid'
            df = self._divide_test_from_valid_by_sub(df)
        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs_std = [ch.upper() for ch in self.config.montage[montage]]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        data = pd.read_csv(file_path, sep=',', header='infer')
        return self._convert_to_mne(data, None)

    def _convert_to_mne(self, data: DataFrame, info) -> mne.io.RawArray:
        ch_names = self.config.montage['10_10']
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.config.orig_fs,
            ch_types=['eeg'] * len(ch_names))

        data.drop(['Time', 'EOG', 'FeedBackEvent'], axis=1, inplace=True)
        signal_data: ndarray = data.to_numpy()
        signal_data = signal_data.astype(np.float32).transpose(1, 0)

        # mne expected unit is Volts, so turn uV to V
        raw = mne.io.RawArray(signal_data / 1e7, info, verbose=False)
        return raw


if __name__ == "__main__":
    builder = InriaBciBuilder('finetune')
    builder.clean_disk_cache()
    builder.preproc()
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
