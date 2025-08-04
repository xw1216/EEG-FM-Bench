import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import numpy as np
import pandas as pd
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGDatasetBuilder, EEGConfig


@dataclass
class TuevConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("2.0.0")
    description: Optional[str] = (
        "This is a subset of the TUH EEG Corpus and contains sessions that are known to contain events "
        "including periodic lateralized epileptiform discharge, generalized periodic epileptiform discharge, "
        "spike and/or sharp wave discharges, artifact, and eye movement. "
        "This version includes fixed edf files that previously had invalid headers which were causing problems.")
    citation: Optional[str] = """\
    @INPROCEEDINGS{7405421,
    author={Harati, A. and Golmohammadi, M. and Lopez, S. and Obeid, I. and Picone, J.},
    booktitle={2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)}, 
    title={Improved EEG event classification using differential energy}, 
    year={2015},
    volume={},
    number={},
    pages={1-4},
    keywords={Electroencephalography;Hidden Markov models;Feature extraction;Brain modeling;Frequency-domain analysis;Mel frequency cepstral coefficient},
    doi={10.1109/SPMB.2015.7405421}}
    """

    filter_notch: float = 60.0
    persist_drop_last: bool = False

    dataset_name: Optional[str] = 'tuev'
    task_type: DatasetTaskType = DatasetTaskType.ARTIFACT
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '01_tcp_ar': [
            'EEG FP1-REF',
            'EEG FP2-REF',
            'EEG F7-REF',
            'EEG F3-REF',
            'EEG FZ-REF',
            'EEG F4-REF',
            'EEG F8-REF',
            'EEG A1-REF',
            'EEG T3-REF',
            'EEG C3-REF',
            'EEG CZ-REF',
            'EEG C4-REF',
            'EEG A2-REF',
            'EEG T4-REF',
            'EEG T5-REF',
            'EEG P3-REF',
            'EEG PZ-REF',
            'EEG P4-REF',
            'EEG T6-REF',
            'EEG O1-REF',
            'EEG O2-REF',
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 5
    suffix_path: str = os.path.join('TUE', 'tuev')
    scan_sub_dir: str = "edf"

    category: list[str] = field(default_factory=lambda: [
        'spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg'])


class TuevBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = TuevConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        split = self._extract_middle_path(file_path, -3, -2)[0]
        file_name = self._extract_file_name(file_path)
        if split == 'train':
            subject, session = file_name.split('_')[-2:]
            session = int(session)
        elif split == 'eval':
            _, subject, _, session = file_name.split('_')
            session = 1 if session == '' else int(session) + 1
        else:
            raise ValueError(f"Invalid split name: {split} when resolve file name.")
        return {
            'subject': subject,
            'session': session,
            'split': split,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            sex = raw.info['subject_info']['sex']
            time = raw.duration

        info.update({
            'montage': '01_tcp_ar',
            'time': time,
            'sex': sex,
        })
        return info

    # noinspection DuplicatedCode
    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        duration = info['time']
        rec_path = file_path[:-3] + 'rec'
        df = pd.read_csv(rec_path, sep=',', header=None, names=['channel', 'start_time', 'stop_time', 'label'])
        df.drop(columns=['channel'], inplace=True)
        # df.drop_duplicates(inplace=True)
        df['label'] = df['label'] - 1

        if len(df.loc[df['start_time'] > duration]) > 0:
            error_df = df.loc[df['start_time'] > duration]
            df.drop(error_df.index, inplace=True)

        if len(df.loc[df['stop_time'] > duration]) > 0:
            df.loc[df['stop_time'] > duration, 'stop_time'] = duration

        df['start_time'] = (df['start_time'] * 1000).astype(int)
        df['stop_time'] = (df['stop_time'] * 1000).astype(int)
        df.reset_index(drop=True, inplace=True)

        # df = df.groupby('label', group_keys=False)[df.columns].apply(self._merge_overlap_labels)
        df.sort_values(by=['start_time', 'stop_time'], inplace=True)
        df = df.reset_index(drop=True)

        annotations = list(zip(np.array(self.config.category)[df['label'].values],
                               df['start_time'].values.tolist(),
                               df['stop_time'].values.tolist()))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        df.loc[df['split'] == 'eval', 'split'] = 'train'

        if self.config.is_finetune:
            df = self._divide_label_balance_all_split(df)
        else:
            df = self._divide_label_balance_all_split(df, splits_name=['train', 'valid'])
        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.split(sep=' ')[1].split('-')[0] for ch in chs]
        chs_std = [self.montage_10_20_replace_dict.get(ch, ch) for ch in chs_std]
        self._std_chs_cache[montage] = chs_std
        return chs_std

if __name__ == "__main__":
    builder = TuevBuilder('finetune')
    # builder.clean_disk_cache()
    # builder.preproc()
    builder.download_and_prepare(num_proc=8)
    dataset = builder.as_dataset()
    print(dataset)

