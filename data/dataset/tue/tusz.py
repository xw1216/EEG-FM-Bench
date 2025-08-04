import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class TuszConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("2.0.3")
    description: Optional[str] = (
        "TUH EEG Seizure Corpus, a corpus developed to motivate the development of "
        "high performance seizure detection algorithms using machine learning. "
        "This corpus is a subset of the TUH EEG Corpus and contains sessions that are "
        "known to contain seizure events. To balance the corpus, some sessions are "
        "provided that do not contain seizure events, so that the false alarm performance "
        "of a system can be tested.")
    citation: Optional[str] = """\
    @ARTICLE{10.3389/fninf.2018.00083,
    AUTHOR={Shah, Vinit  and von Weltin, Eva  and Lopez, Silvia  and McHugh, James Riley  and Veloso, Lillian  and Golmohammadi, Meysam  and Obeid, Iyad  and Picone, Joseph },
    TITLE={The Temple University Hospital Seizure Detection Corpus},
    JOURNAL={Frontiers in Neuroinformatics},
    VOLUME={12},
    YEAR={2018},
    URL={https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00083},
    DOI={10.3389/fninf.2018.00083},
    ISSN={1662-5196}}
    """

    filter_notch: float = 60.0
    persist_drop_last: bool = False

    dataset_name: Optional[str] = 'tusz'
    task_type: DatasetTaskType = DatasetTaskType.SEIZURE
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '01_tcp_ar': [
            'EEG FP1-REF',
            'EEG FP2-REF',
            'EEG F7-REF',
            'EEG F3-REF',
            # 'EEG FZ-REF',
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
            # 'EEG PZ-REF',
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
            # 'EEG FZ-LE',
            'EEG F4-LE',
            'EEG F8-LE',
            'EEG A1-LE',
            'EEG T3-LE',
            'EEG C3-LE',
            'EEG CZ-LE',
            'EEG C4-LE',
            'EEG T4-LE',
            'EEG A2-LE',
            'EEG T5-LE',
            'EEG P3-LE',
            # 'EEG PZ-LE',
            'EEG P4-LE',
            'EEG T6-LE',
            'EEG O1-LE',
            'EEG OZ-LE',
            'EEG O2-LE',
        ],
        '03_tcp_ar_a': [
            'EEG FP1-REF',
            'EEG FP2-REF',
            'EEG F7-REF',
            'EEG F3-REF',
            # 'EEG FZ-REF',
            'EEG F4-REF',
            'EEG F8-REF',
            'EEG T3-REF',
            'EEG C3-REF',
            'EEG CZ-REF',
            'EEG C4-REF',
            'EEG T4-REF',
            'EEG T5-REF',
            'EEG P3-REF',
            # 'EEG PZ-REF',
            'EEG P4-REF',
            'EEG T6-REF',
            'EEG O1-REF',
            'EEG O2-REF',
        ],
    })

    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    wnd_div_sec: int = 10
    suffix_path: str = os.path.join('TUE', 'tusz')
    scan_sub_dir: str = "edf"

    category: list[str] = field(default_factory=lambda: ['seiz', 'bckg'])
    is_binary_class: bool = True


class TuszBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = TuszConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True, is_binary_class=True),
        BUILDER_CONFIG_CLASS(
            name='finetune-multiple', is_finetune=True, is_binary_class=False,
            category=[
                'null', 'spsw', 'gped', 'pled', 'eybl', 'artf', 'bckg',
                'seiz', 'fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz',
                'cnsz', 'tcsz', 'atsz', 'mysz', 'nesz', 'intr', 'slow',
                'eyem', 'chew', 'shiv', 'musc', 'elpp', 'elst', 'calb',
                'hphs', 'trip',
            ]
        ),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self. _extract_file_name(file_path)
        subject, session, term = file_name.split('_')[-3:]
        session = int(session[1:])
        term = int(term[1:])
        return {
            'subject': subject,
            'session': session,
            'term': term,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        split, _, _, montage = self._extract_middle_path(file_path, -5, -1)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            sex = raw.info['subject_info']['sex']
            time = raw.duration

        info.update({
            'split': split,
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
        if self.config.is_binary_class:
            tim_path = file_path[:-3] + 'csv_bi'
        else:
            tim_path = file_path[:-3] + 'csv'

        df = pd.read_csv(tim_path, sep=',', comment='#')
        df.drop(columns=['channel'], inplace=True)
        df.drop(df[df['confidence'] < 0.5].index, inplace=True)
        df.drop_duplicates(inplace=True)

        if len(df.loc[df['start_time'] > duration]) > 0:
            error_df = df.loc[df['start_time'] > duration]
            df.drop(error_df.index, inplace=True)

        if len(df.loc[df['stop_time'] > duration]) > 0:
            df.loc[df['stop_time'] > duration, 'stop_time'] = duration

        df['start_time'] = (df['start_time'] * 1000).astype(int)
        df['stop_time'] = (df['stop_time'] * 1000).astype(int)
        df.reset_index(drop=True, inplace=True)

        # samples will be less if do overlap merging
        # df = df.groupby('label', group_keys=False)[df.columns].apply(self._merge_overlap_labels)
        df.sort_values(by=['start_time', 'stop_time'], inplace=True)
        df = df.reset_index(drop=True)

        annotations = list(zip(
            df['label'].values,
            df['start_time'].values.tolist(),
            df['stop_time'].values.tolist()
        ))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        df.loc[df['split'] == 'dev', 'split'] = 'valid'
        df.loc[df['split'] == 'eval', 'split'] = 'test'
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
    builder = TuszBuilder('finetune')
    # builder.clean_disk_cache()
    # builder.preproc()
    builder.download_and_prepare(num_proc=8)
    dataset = builder.as_dataset()
    print(dataset)


