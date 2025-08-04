import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
import pandas as pd
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class SienaScalpConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "The database consists of EEG recordings of 14 patients acquired at the Unit of Neurology and "
        "Neurophysiology of the University of Siena.  Subjects include 9 males (ages 25-71) and 5 females "
        "(ages 20-58). Subjects were monitored with a Video-EEG with a sampling rate of 512 Hz, with "
        "electrodes arranged on the basis of the international 10-20 System. Most of the recordings also "
        "contain 1 or 2 EKG signals. The diagnosis of epilepsy and the classification of seizures according "
        "to the criteria of the International League Against Epilepsy were performed by an expert clinician "
        "after a careful review of the clinical and electrophysiological data of each patient.")
    citation: Optional[str] = """\
    @Article{pr8070846,
    AUTHOR = {Detti, Paolo and Vatti, Giampaolo and Zabalo Manrique de Lara, Garazi},
    TITLE = {EEG Synchronization Analysis for Seizure Prediction: A Study on Data of Noninvasive Recordings},
    JOURNAL = {Processes},
    VOLUME = {8},
    YEAR = {2020},
    NUMBER = {7},
    ARTICLE-NUMBER = {846},
    URL = {https://www.mdpi.com/2227-9717/8/7/846},
    ISSN = {2227-9717},
    DOI = {10.3390/pr8070846}}
    """

    filter_notch: float = 50.0
    persist_drop_last: bool = False

    dataset_name: Optional[str] = 'siena_scalp'
    task_type: DatasetTaskType = DatasetTaskType.SEIZURE
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                       'FP1',       'FP2',
            'F9', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F10',
                'FC5', 'FC1',       'FC2', 'FC6',
                  'T3', 'C3', 'CZ', 'C4', 'T4',
                'CP5', 'CP1',       'CP2', 'CP6',
                  'T5', 'P3', 'PZ', 'P4', 'T6',
                        'O1',       'O2',
    ]
    })

    # Bad channels in PN-10
    # ['Cp1', 'Cp2', 'Cp5', 'Cp6', 'F10', 'F9', 'Fc1', 'Fc2', 'Fc5', 'Fc6']

    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    wnd_div_sec: int = 10
    suffix_path: str = "Siena Scalp EEG Dataset"
    scan_sub_dir: str = "siena-scalp-eeg-database-1.0.0"

    category: list[str] = field(default_factory=lambda: ['seizure', 'normal'])


class SienaScalpBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = SienaScalpConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()


    def _load_meta_info(self):
        sub_meta = [
            'PN00-1', 55, 'Male', 70773, 71916, 71986,
            'PN00-2', 55, 'Male', 8297, 9517, 9571,
            'PN00-3', 55, 'Male', 65744, 66509, 66569,
            'PN00-4', 55, 'Male', 75103, 76109, 76183,
            'PN00-5', 55, 'Male', 80524, 81428, 81495,
            'PN01-1', 46, 'Male', 68444, 78662, 78716,
            'PN01-1', 46, 'Male', 68444, 114797, 114871,
            'PN03-1', 54, 'Male', 81877, 120550, 120661,
            'PN03-2', 54, 'Male', 77464, 112385, 112518,
            'PN05-2', 51, 'Female', 24362, 31525, 31560,
            'PN05-3', 51, 'Female', 21683, 28519, 28549,
            'PN05-4', 51, 'Female', 23915, 27523, 27562,
            'PN06-1', 36, 'Male', 15682, 21265, 21329,
            'PN06-2', 36, 'Male', 76289, 85149, 85218,
            'PN06-3', 36, 'Male', 23151, 29426, 29468,
            'PN06-4', 36, 'Male', 40569, 46508, 46571,
            'PN06-5', 36, 'Male', 48281, 53064, 53108,
            'PN07-1', 20, 'Female', 83890, 105949, 106011,
            'PN09-1', 27, 'Female', 50934, 58183, 58263,
            'PN09-2', 27, 'Female', 54129, 61256, 61315,
            'PN09-3', 27, 'Female', 51623, 58844, 58908,
            'PN10-1', 25, 'Male', 20405, 27950, 28019,
            'PN10-2', 25, 'Male', 34215, 42013, 42064,
            'PN10-3', 25, 'Male', 48798, 56639, 56702,
            'PN10-4.5.6', 25, 'Male', 43881, 46190, 46195,
            'PN10-4.5.6', 25, 'Male', 43881, 50425, 50444,
            'PN10-4.5.6', 25, 'Male', 43881, 55106, 55163,
            'PN10-7.8.9', 25, 'Male', 60565, 63313, 63361,
            'PN10-7.8.9', 25, 'Male', 60565, 66024, 66042,
            'PN10-7.8.9', 25, 'Male', 60565, 73488, 73503,
            'PN10-10', 25, 'Male', 31522, 39499, 39513,
            'PN11-1', 58, 'Female', 41485, 49039, 49094,
            'PN12-1.2', 71, 'Male', 57091, 58403, 58466,
            'PN12-1.2', 71, 'Male', 57091, 66661, 66729,
            'PN12-3', 71, 'Male', 31355, 32127, 32223,
            'PN12-4', 71, 'Male', 57559, 67371, 67434,
            'PN13-1', 34, 'Female', 30268, 37330, 37378,
            'PN13-2', 34, 'Female', 24902, 32151, 32216,
            'PN13-3', 34, 'Female', 43201, 50754, 50905,
            'PN14-1', 49, 'Male', 42298, 49560, 49587,
            'PN14-2', 49, 'Male', 57013, 64492, 64504,
            'PN14-3', 49, 'Male', 58665, 76205, 76246,
            'PN14-4', 49, 'Male', 51510, 56973, 57056,
            'PN16-1', 41, 'Female', 74721, 81905, 82028,
            'PN16-2', 41, 'Female', 3235, 11809, 11916,
            'PN17-1', 42, 'Male', 72868, 81288, 81358,
            'PN17-2', 42, 'Male', 49938, 57669, 57752,
        ]

        arr = np.array(sub_meta).reshape(-1, 6)

        # 创建 DataFrame
        df = pd.DataFrame(arr, columns=[
            'name', 'age', 'gender', 'start_time', 'seiz_start', 'seiz_end'
        ])

        # 类型转换（如果需要把数字列变成 int）
        df['age'] = df['age'].astype(int)
        df['start_time'] = df['start_time'].astype(int)
        df['seiz_start'] = df['seiz_start'].astype(int)
        df['seiz_end'] = df['seiz_end'].astype(int)

        self.sub_meta = df

    @staticmethod
    def _extract_file_name(file_path: str):
        return os.path.basename(file_path)[:-4]

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, session = file_name.split('-')
        subject = subject[2:]
        return {
            'subject': int(subject),
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        age, sex = self.sub_meta.loc[self.sub_meta['name'] == file_name, ['age', 'gender']].values.tolist()[0]
        sex = sex[0]

        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as data:
            time = data.times[-1] + data.times[1]

        info.update({
            'montage': '10_20',
            'age': age,
            'sex': sex,
            'time': time,
        })
        return info

    def get_non_seizure_intervals(self, name, total_end):
        subset = self.sub_meta[self.sub_meta['name'] == name]
        if subset.empty:
            return [(0, total_end)]

        subset = subset[['start_time', 'seiz_start', 'seiz_end']]

        # noinspection PyArgumentList
        subset = subset.sort_values(by='seiz_start')

        non_seizure_intervals = []
        current = 0

        for _, row in subset.iterrows():
            start_time = row['start_time']
            seiz_start = row['seiz_start'] - start_time
            seiz_end = row['seiz_end'] - start_time

            # 如果当前时间在 seizure 前，说明中间有空白
            if current < seiz_start:
                non_seizure_intervals.append((current, seiz_start))

            # 移动 current 到 seizure 之后
            current = max(current, seiz_end)

        # 最后是否还有结尾部分
        if current < total_end and (total_end - current) > self.config.wnd_div_sec:
            non_seizure_intervals.append((current, total_end))

        return non_seizure_intervals

    # noinspection PyTypeChecker
    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        file_name = self._extract_file_name(file_path)
        matched_rows = self.sub_meta[self.sub_meta['name'] == file_name]

        annotations = []
        for idx, row in matched_rows.iterrows():
            start_time = row['start_time']
            rel_start: int = row['seiz_start'] - start_time
            rel_end: int = row['seiz_end'] - start_time

            assert rel_end < info['time']
            annotations.append((
                self.config.category[0],
                int(rel_start * 1000),
                int(rel_end * 1000),
            ))

        normal_intervals: list = self.get_non_seizure_intervals(file_name, info['time'])

        for row in normal_intervals:
            annotations.append((
                self.config.category[1],
                int(row[0] * 1000),
                int(row[1] * 1000),
            ))

        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        if not self.config.is_finetune:
            return self._divide_all_split_by_sub(df)

        df['split'] = 'train'
        df.loc[df['subject'].isin([16, 17]), 'split'] = 'test'

        # name 中包含 PN13 或 PN14 的设为 'valid'
        df.loc[df['subject'].isin([9, 13]), 'split'] = 'valid'
        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [self.montage_10_20_replace_dict.get(ch, ch) for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_edf(file_path, verbose=verbose, preload=preload)

            channel_mapping = {}
            for ch in raw.ch_names:
                name = ch.upper()
                if name.startswith('EEG'):
                    name = name[4:]
                channel_mapping[ch] = name

            raw.rename_channels(channel_mapping)
            raw.pick(self.config.montage['10_20'])

            dig_montage = mne.channels.make_standard_montage('standard_1020')
            mapping = {old_name: old_name.upper() for old_name in dig_montage.ch_names}
            dig_montage.rename_channels(mapping)
            raw.set_montage(dig_montage)

            if 'PN10-' in file_path and preload:
                raw.info['bads'].extend(['CP1', 'CP2', 'CP5', 'CP6', 'F10', 'F9', 'FC1', 'FC2', 'FC5', 'FC6', 'PZ'])
                raw = raw.interpolate_bads(reset_bads=True)

            return raw


if __name__ == "__main__":
    builder = SienaScalpBuilder('finetune')
    builder.preproc(n_proc=2)
    builder.download_and_prepare(num_proc=2)
    dataset = builder.as_dataset()
    print(dataset)
