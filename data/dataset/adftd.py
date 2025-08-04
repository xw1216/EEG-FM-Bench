import logging
import os.path
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne_bids
import pandas as pd
from mne.io import BaseRaw
from mne_bids import get_entity_vals, BIDSPath
from pandas import DataFrame
from tqdm import tqdm

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class AdftdConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.8")
    description: Optional[str] = (
        "This dataset contains the EEG resting state-closed eyes recordings from 88 subjects in total."
        "36 of them were diagnosed with Alzheimer's disease (AD group), 23 were diagnosed with Frontotemporal "
        "Dementia (FTD group) and 29 were healthy subjects (CN group). Cognitive and neuropsychological state "
        "was evaluated by the international Mini-Mental State Examination (MMSE). MMSE score ranges from 0 to 30, "
        "with lower MMSE indicating more severe cognitive decline. Recordings were aquired from the 2nd Department of "
        "Neurology of AHEPA General Hospital of Thessaloniki by an experienced team of neurologists. For recording, a "
        "Nihon Kohden EEG 2100 clinical device was used, with 19 scalp electrodes according to the 10-20 international "
        "system and 2 reference electrodes (A1 and A2) placed on the mastoids for impendance check, according to the "
        "manual of the device. The sampling rate was 500 Hz with 10uV/mm resolution.")
    citation: Optional[str] = """\
    @misc{miltiadousdataset,
    title={A dataset of scalp EEG recordings of Alzheimer’s disease, frontotemporal dementia and healthy subjects from 
    routine EEG. Data 8 (6)(2023)},
    author={Miltiadous, A and Tzimourta, KD and Afrantou, T and Ioannidis, P and Grigoriadis, N and Tsalikakis, DG and 
    Angelidis, P and Tsipouras, MG and Glavas, E and Giannakeas, N and others}
    }
    """

    filter_notch: float = 50.0
    is_notched: bool = True

    dataset_name: Optional[str] = 'adftd'
    task_type: DatasetTaskType = DatasetTaskType.CLINICAL
    file_ext: str = 'set'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                    'Fp1', 'Fp2',
            'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
                     'O1', 'O2',
        ]
    })

    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    wnd_div_sec: int = 10
    suffix_path: str = 'ADFTD'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: ['AD', 'FTD', 'CN'])


class AdftdBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = AdftdConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        df = pd.read_csv(
            os.path.join(self.config.raw_path, self.config.scan_sub_dir, 'participants.tsv'), sep='\t')

        self.sub_meta: DataFrame = df

    def _walk_raw_data_files(self):
        # !IMPORTANT derivatives folder should be moved out from the scan folder to avoid duplicate files
        raw_data_files = []
        logger.info('Parsing BIDS path...')
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        for subject in tqdm(get_entity_vals(scan_path, "subject"),
                            desc='Searching file system', unit='sub'):
            bids_path = BIDSPath(subject=subject, datatype="eeg", root=scan_path, extension=".set")
            matched_paths = bids_path.match()
            for path in matched_paths:
                raw_data_files.append(str(path.fpath))

        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        bids_path = mne_bids.get_bids_path_from_fname(file_path)
        return {
            'subject': bids_path.subject,
            'session': 1,
            'task': bids_path.task,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        row = self.sub_meta[self.sub_meta['participant_id'] == f'sub-{info["subject"]}'].iloc[0]
        sex, age, group = row['Gender'], row['Age'].item(), row['Group']
        montage = '10_20'

        with self._read_raw_data(file_path) as raw:
            time = raw.duration

        info.update({
            'montage': montage,
            'time': time,
            'sex': sex,
            'age': age,
            'group': group,
        })

        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        group = info['group']
        if group == 'A':
            return [('AD', 0, -1)]
        elif group == 'F':
            return [('FTD', 0, -1)]
        elif group == 'C':
            return [('CN', 0, -1)]
        else:
            raise ValueError(f'Invalid group: {group}')

    def _divide_split(self, df: DataFrame) -> DataFrame:
        df = self._divide_label_balance_all_split(df, None if self.config.is_finetune else ['train', 'valid'])
        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs: list[str] = self.config.montage[montage]
        chs_std = [ch.upper() for ch in chs]
        chs_std = [self.montage_10_20_replace_dict.get(ch, ch) for ch in chs_std]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            bids_path = mne_bids.get_bids_path_from_fname(file_path)
            raw = mne_bids.read_raw_bids(bids_path, verbose=verbose)
            return raw


if __name__ == "__main__":
    builder = AdftdBuilder("finetune")
    builder.clean_disk_cache()
    builder.preproc(n_proc=1)
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
