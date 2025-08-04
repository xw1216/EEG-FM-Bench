import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import mne_bids
import pandas as pd
from mne.io import BaseRaw
from mne_bids import BIDSPath
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class ThingsEEGConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "THINGS-EEG, a dataset containing human electroencephalography responses from 50 subjects to 1,854 object "
        "concepts and 22,248 images in the THINGS stimulus set, a manually curated and high-quality image database "
        "that was specifically designed for studying human vision. The THINGS-EEG dataset provides neuroimaging "
        "recordings to a systematic collection of objects and concepts and can therefore support a wide array of "
        "research to understand visual object processing in the human brain.")
    citation: Optional[str] = """\
    @Article{Grootswagers2022,
    author={Grootswagers, Tijl and Zhou, Ivy and Robinson, Amanda K. and Hebart, Martin N. and Carlson, Thomas A.},
    title={Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams},
    journal={Scientific Data},
    year={2022},
    month={Jan},
    day={10},
    volume={9},
    number={1},
    pages={3},
    issn={2052-4463},
    doi={10.1038/s41597-021-01102-7},
    url={https://doi.org/10.1038/s41597-021-01102-7}
    }
    """

    filter_notch: float = 50.0

    dataset_name: Optional[str] = 'things_eeg'
    task_type: DatasetTaskType = DatasetTaskType.VISUAL
    file_ext: str = 'vhdr'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                                            'Fp1', 'Fp2',
                                 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
                        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
                           'T7', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'T8',
            'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
                        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                                 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                          'O1', 'Oz', 'O2',
        ]
    })

    valid_ratio: float = 0.20
    test_ratio: float = 0.0
    wnd_div_sec: int = 20
    suffix_path: str = 'THINGS-EEG'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: [])


class ThingsEEGBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = ThingsEEGConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        participant_csv = os.path.join(self.config.raw_path, self.config.scan_sub_dir, 'participants.tsv')
        self.sub_meta = pd.read_csv(participant_csv, sep='\t')

    def _walk_raw_data_files(self):
        # !IMPORTANT derivatives folder should be moved out from the scan folder to avoid duplicate files
        logger.info('Parsing BIDS path...')
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        bids_path = BIDSPath(datatype="eeg", root=scan_path, suffix='eeg', extension=f".{self.config.file_ext}")
        all_paths = bids_path.match()

        raw_data_files = [str(path.fpath) for path in all_paths]
        return raw_data_files


    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        bids_path = mne_bids.get_bids_path_from_fname(file_path)
        return {
            'subject': bids_path.subject,
            'session': 1,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        montage = '10_20'

        row = self.sub_meta[self.sub_meta['participant_id'] == f"sub-{info['subject']}"].iloc[0]
        age = row['age']
        sex = row['gender']

        with self._read_raw_data(file_path) as raw:
            time = raw.duration

        info.update({
            'montage': montage,
            'time': time,
            'age': age,
            'sex': sex,
        })

        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        return [('default', 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs: list[str] = self.config.montage[montage]
        chs_std = [ch.upper() for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        raw = mne.io.read_raw_brainvision(file_path, preload=preload, verbose=verbose)
        return raw

if __name__ == "__main__":
    builder = ThingsEEGBuilder("pretrain")
    builder.clean_disk_cache()
    builder.preproc(n_proc=1)
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
