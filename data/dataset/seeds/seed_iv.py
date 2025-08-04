import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
import pandas as pd
import pytz
import s3fs
from mne.io import BaseRaw
from numpy import ndarray
from pandas import DataFrame
from scipy.io import loadmat

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class SeedIVConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "SJTU Emotion EEG Dataset for Four Emotions (SEED-IV) Seventy-two film clips were carefully "
        "chosen by a preliminary study, which had the tendency to induce happiness, sadness, fear "
        "or neutral emotions. A total of 15 subjects participated in the experiment. For each participant, "
        "3 sessions were performed on different days, and each session contained 24 trials. In one trial, "
        "the participant watched one of the film clips, while his or her EEG signals and eye movements "
        "were collected with the 62-channel")
    citation: Optional[str] = """\
    @ARTICLE{8283814, 
    author={W. Zheng and W. Liu and Y. Lu and B. Lu and A. Cichocki}, 
    journal={IEEE Transactions on Cybernetics}, 
    title={EmotionMeter: A Multimodal Framework for Recognizing Human Emotions}, 
    year={2018}, 
    volume={}, 
    number={}, 
    pages={1-13}, 
    keywords={Electroencephalography;Emotion recognition;Electrodes;Feature extraction;Human computer interaction;Biological neural networks;Brain modeling;Affective brain-computer interactions;deep learning;EEG;emotion recognition;eye movements;multimodal deep neural networks}, 
    doi={10.1109/TCYB.2018.2797176}, 
    ISSN={2168-2267}, 
    month={},}
    """

    filter_notch: float = 50.0
    orig_fs: float = 200.0

    dataset_name: Optional[str] = 'seed_iv'
    task_type: DatasetTaskType = DatasetTaskType.EMOTION
    file_ext: str = 'mat'
    montage: dict[str, list[str]] = field(default_factory=lambda :{
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
    remove_ch: dict[str, int] = field(default_factory=lambda: {'CB1': 57, 'CB2': 61})

    valid_ratio: float = 0.067
    test_ratio: float = 0.067
    wnd_div_sec: int = 10
    suffix_path: str = os.path.join('SEED', 'SEED_IV')
    scan_sub_dir: str = "eeg_raw_data"

    category: list[str] = field(default_factory=lambda: ['neutral', 'sad', 'fear', 'happy'])


class SeedIVBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = SeedIVConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()
        self.n_trial = self.label_meta.shape[1]

    def _load_meta_info(self):
        self.sub_meta = pd.DataFrame({
            'subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ],
            'sex': ['M', 'M', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'F', 'F', ]
        })

        self.label_meta: ndarray = np.array(
            [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
             [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
             [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
        )

    def _persist_example_file(self, sample: dict):
        # pretrain datasets have no ground truth will be assigned a label item which indicates all signal array
        path, montage, labels, split = (
            sample['path'], sample['montage'], json.loads(sample['label']), sample['split'])
        mid_df = pd.DataFrame(columns=[
            'key', 'split', 'cnt'])

        try:
            (data_list, trial_names) = self._read_raw_data(path, preload=True, verbose=False)
            for data, label, trial in zip(data_list, labels, trial_names):
                log_path = f"{path}_trial_{trial}"
                data = self._select_data_channels(data, path, montage)
                data = self._resample_and_filter(data)
                raw = self._fetch_signal_ndarray(data)
                chs_idx = self._fetch_chs_index(montage)

                examples = self._generate_window_sample(raw, montage, chs_idx, [label], self.config.persist_drop_last)
                df = pd.DataFrame(data=examples)
                filename = f"{self._encode_path(log_path)}.parquet"
                output_path = self._build_output_dir(split, filename)

                if self.config.is_remote_fs:
                    fs = s3fs.S3FileSystem(**self.s3_conf)
                    with fs.open(output_path, 'wb') as f:
                        df.to_parquet(
                            f, engine='pyarrow', index=False,
                            compression=self.config.mid_compress_algo,)
                    fs.invalidate_cache()
                else:
                    df.to_parquet(
                        output_path, engine='pyarrow', index=False,
                        compression=self.config.mid_compress_algo,)
                row = {
                    'key': filename,
                    'split': split,
                    'cnt': len(examples)}
                mid_df.loc[len(mid_df)] = row
        except Exception as e:
            logger.error(f"Error persisting example file {path}: {str(e)}")
            return None

        return mid_df

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, date = file_name.split('_')
        data_obj = datetime.strptime(date, "%Y%m%d")
        time_zone = pytz.timezone('Asia/Shanghai')
        data_obj = time_zone.localize(data_obj)
        return {
            'subject': int(subject),
            'date': data_obj.strftime("%Y%m%d"),
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        session = self._extract_middle_path(file_path, -2, -1)[0]
        sex = self.sub_meta.loc[info['subject'] - 1, 'sex']
        sex = 1 if sex == 'M' else 2

        time = []
        data = loadmat(file_path)
        keys = [key for key in data.keys() if '__' not in key]

        assert len(keys) == self.n_trial
        for trial_name in sorted(keys, key=lambda x: int(x.split('eeg')[1])):
            time.append(data[trial_name].shape[1] / self.config.orig_fs)

        info.update({
            'session': int(session),
            'montage': '10_10',
            'sex': sex,
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        session = info['session']
        labels = self.label_meta[session - 1]

        annotations = []
        for label in labels:
            annotations.append((self.config.category[label],0,-1))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

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

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        data = loadmat(file_path)
        trial_names = sorted(
            [key for key in data.keys() if '__' not in key],
            key=lambda x: int(x.split('eeg')[1]))

        raw_list = []
        for j, trial in enumerate(trial_names):
            trial_data = self._convert_to_mne(data[trial], None)
            raw_list.append(trial_data)
        return raw_list, trial_names

    def _convert_to_mne(self, data: ndarray, info):
        ch_names = self._orig_ch_names()
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.config.orig_fs,
            ch_types=['eeg'] * len(ch_names))

        # mne expected unit is Volts, so turn uV to V
        raw = mne.io.RawArray(data / 1e6, info, verbose=False)
        return raw


if __name__ == "__main__":
    builder = SeedIVBuilder('pretrain')
    builder.preproc()
    builder.download_and_prepare(num_proc=4)
    dataset = builder.as_dataset()
    print(dataset)
