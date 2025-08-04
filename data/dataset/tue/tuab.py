import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGDatasetBuilder, EEGConfig


@dataclass
class TuabConfig(EEGConfig):
    name: str = "pretrain"
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("3.0.1")
    description: Optional[str] = (
        'This dataset is a subset of the TUH EEG Abnormal Corpus, '
        'which contains EEG records that are classified as clinically normal or abnormal.')
    citation: Optional[str] = """\
    @INPROCEEDINGS{7405423,
    author={López, S. and Suarez, G. and Jungreis, D. and Obeid, I. and Picone, J.},
    booktitle={2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)}, 
    title={Automated identification of abnormal adult EEGs}, 
    year={2015},
    volume={},
    number={},
    pages={1-5},
    keywords={Electroencephalography;Radio frequency;Principal component analysis;Algorithm design and analysis;Vegetation;Training;Rhythm},
    doi={10.1109/SPMB.2015.7405423}}
    """

    filter_notch: float = 60.0

    dataset_name: Optional[str] = 'tuab'
    task_type: DatasetTaskType = DatasetTaskType.CLINICAL
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
            'EEG T1-REF',
            'EEG T2-REF'
        ]
    })

    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    wnd_div_sec: int = 30
    suffix_path: str =  os.path.join('TUE', 'tuab')
    scan_sub_dir: str = 'edf'

    category: list[str] = field(default_factory=lambda: ['normal', 'abnormal'])


class TuabBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = TuabConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain', wnd_div_sec=30),
        BUILDER_CONFIG_CLASS(name='finetune', wnd_div_sec=15, is_finetune=True)
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
        split, _, montage = self._extract_middle_path(file_path, -4, -1)
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

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]) -> dict[str, list[tuple[str, int, int]]]:
        label = self._extract_middle_path(file_path, -3, -2)[0]
        return [(label, 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        df.loc[df['split'] == 'eval', 'split'] = 'valid'
        return self._divide_balance_test_from_valid(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.split(sep=' ')[1].split('-')[0] for ch in chs]
        chs_std = [self.montage_10_20_replace_dict.get(ch, ch) for ch in chs_std]
        self._std_chs_cache[montage] = chs_std
        return chs_std

if __name__ == "__main__":
    builder = TuabBuilder('finetune')
    # builder.clean_disk_cache()
    # builder.preproc()
    builder.download_and_prepare(num_proc=4)
    dataset = builder.as_dataset()
    print(dataset)

