import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import pandas as pd
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class TuarConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("3.0.1")
    description: Optional[str] = (
        "The TUH EEG Artifact (TUAR) Corpus began as an effort to identify "
        "artifacts that could be used to train artifact models. "
        "In annotating v1.0.0, we identified a suitable number of events "
        "for each type of artifact. We did not annotate the entire signal.")
    citation: Optional[str] = """\
    @INPROCEEDINGS{9672302,
    author={Buckwalter, G. and Chhin, S. and Rahman, S. and Obeid, I. and Picone, J.},
    booktitle={2021 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)}, 
    title={Recent Advances in the TUH EEG Corpus: Improving the Interrater Agreement for Artifacts and Epileptiform Events}, 
    year={2021},
    volume={},
    number={},
    pages={1-3},
    keywords={Hospitals;Machine learning;Signal processing;Electroencephalography;Biology;Task analysis},
    doi={10.1109/SPMB52430.2021.9672302}}
    """

    filter_notch: float = 60.0
    persist_drop_last: bool = False

    dataset_name: Optional[str] = 'tuar'
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
            'EEG T3-REF',
            'EEG C3-REF',
            'EEG CZ-REF',
            'EEG C4-REF',
            'EEG T4-REF',
            'EEG T5-REF',
            'EEG P3-REF',
            'EEG PZ-REF',
            'EEG P4-REF',
            'EEG T6-REF',
            'EEG O1-REF',
            'EEG O2-REF',
        ],
        '02_tcp_le': [
            'EEG FP1-LE',
            'EEG FP2-LE',
            'EEG F7-LE',
            'EEG F3-LE',
            'EEG FZ-LE',
            'EEG F4-LE',
            'EEG F8-LE',
            'EEG T3-LE',
            'EEG C3-LE',
            'EEG CZ-LE',
            'EEG C4-LE',
            'EEG T4-LE',
            'EEG T5-LE',
            'EEG P3-LE',
            'EEG PZ-LE',
            'EEG P4-LE',
            'EEG T6-LE',
            'EEG O1-LE',
            'EEG O2-LE',
        ],
        '03_tcp_ar_a': [
            'EEG FP1-REF',
            'EEG FP2-REF',
            'EEG F7-REF',
            'EEG F3-REF',
            'EEG FZ-REF',
            'EEG F4-REF',
            'EEG F8-REF',
            'EEG T3-REF',
            'EEG C3-REF',
            'EEG CZ-REF',
            'EEG C4-REF',
            'EEG T4-REF',
            'EEG T5-REF',
            'EEG P3-REF',
            'EEG PZ-REF',
            'EEG P4-REF',
            'EEG T6-REF',
            'EEG O1-REF',
            'EEG O2-REF',
        ],
    })
    unify_montage: str = '10_20'

    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    wnd_div_sec: int = 5
    suffix_path: str = os.path.join('TUE', 'tuar')
    scan_sub_dir: str = "edf"

    # Other artifacts whose percentage is less than 0.5% is excluded
    category: list[str] = field(default_factory=lambda: [
        'musc', 'eyem', 'elec', 'chew',
        'eyem_musc', 'musc_elec', 'eyem_elec', 'eyem_chew'])


class TuarBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = TuarConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, session, term = file_name.split('_')
        session = int(session[1:])
        term = int(term[1:])
        return {
            'subject': subject,
            'session': session,
            'term': term,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        montage = self._extract_middle_path(file_path, -2, -1)[0]
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            sex = raw.info['subject_info']['sex']
            time = raw.duration

        info.update({
            'montage': montage,
            'time': time,
            'sex': sex,
        })
        return info

    # noinspection DuplicatedCode
    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        duration = info['time']

        tim_path = file_path[:-3] + 'csv'
        df = pd.read_csv(tim_path, sep=',', comment='#')
        df.drop(df[df['confidence'] < 0.5].index, inplace=True)
        df.drop(columns=['channel'], inplace=True)
        df.drop_duplicates(inplace=True)

        if len(df.loc[df['start_time'] > duration]) > 0:
            error_df = df.loc[df['start_time'] > duration]
            df.drop(error_df.index, inplace=True)

        if len(df.loc[df['stop_time'] > duration]) > 0:
            df.loc[df['stop_time'] > duration, 'stop_time'] = duration

        df['start_time'] = (df['start_time'] * 1000).astype(int)
        df['stop_time'] = (df['stop_time'] * 1000).astype(int)
        df.reset_index(drop=True, inplace=True)

        df = df.groupby('label', group_keys=False)[df.columns].apply(self._merge_overlap_labels)
        df.sort_values(by=['start_time', 'stop_time'], inplace=True)
        df = df.reset_index(drop=True)

        annotations = list(zip(
            df['label'].values.tolist(),
            df['start_time'].values.tolist(),
            df['stop_time'].values.tolist()
        ))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        df = self._divide_label_balance_all_split(df, None if self.config.is_finetune else ['train', 'valid'])
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
    builder = TuarBuilder('finetune')
    builder.clean_disk_cache()
    builder.preproc()
    builder.download_and_prepare(
        num_proc=8,
    )
    dataset = builder.as_dataset()
    print(dataset)
