import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne.io
import numpy as np
import pandas as pd
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class BrainLatConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "The Latin American Brain Health Institute (BrainLat) dataset comprises multimodal neuroimaging data of 780 "
        "participants from Latin American. The dataset includes 530 patients with neurodegenerative diseases such "
        "as Alzheimer's disease (AD), behavioral variant frontotemporal dementia (bvFTD), multiple sclerosis (MS), "
        "Parkinson's disease (PD), and 250 healthy controls (HCs). This dataset (62.7±9.5 years, age range "
        "21–89 years) was collected through a multicentric effort across five Latin American countries to address "
        "the need for affordable, scalable, and available biomarkers in regions with larger inequities. The BrainLat "
        "dataset is the first regional collection of clinical and cognitive assessments, anatomical magnetic resonance "
        "imaging (MRI), resting-state functional MRI (fMRI), diffusion-weighted MRI (DWI), and high density "
        "resting-state electroencephalography (EEG) in dementia patients. In addition, it includes demographic "
        "information about harmonized recruitment and assessment protocols. The dataset is publicly available to "
        "encourage further research and development of tools and health applications for neurodegeneration based "
        "on multimodal neuroimaging, promoting the assessment of regional variability and inclusion of underrepresented "
        "participants in research. The BrainLat dataset contains neuroimaging and cognitive data from 780 subjects, "
        "including patients with AD (N =278), bvFTD (N =163), PD (N =57) and MS (N =32), and HCs (N =250).")
    citation: Optional[str] = """\
    @article{prado2023brainlat,
      title={The BrainLat project, a multimodal neuroimaging dataset of neurodegeneration from underrepresented backgrounds},
      author={Prado, Pavel and Medel, Vicente and Gonzalez-Gomez, Raul and Sainz-Ballesteros, Agust{\'\i}n and Vidal, Victor and Santamar{\'\i}a-Garc{\'\i}a, Hernando and Moguilner, Sebastian and Mejia, Jhony and Slachevsky, Andrea and Behrens, Maria Isabel and others},
      journal={Scientific Data},
      volume={10},
      number={1},
      pages={889},
      year={2023},
      publisher={Nature Publishing Group UK London}
    }
    """

    filter_notch: float = 60.0
    is_notched: bool = True

    dataset_name: Optional[str] = 'brain_lat'
    task_type: DatasetTaskType = DatasetTaskType.CLINICAL
    file_ext: str = 'set'

    # The corresponding electrodes between standard 10-10 and biosemi montage is calculated by
    # mne montage digital position in Munkres algorithm with max_distance=0.01.
    # The layout of the biosemi montage is shown in the figure below:
    # https://www.biosemi.com/pics/cap_128_layout_medium.jpg
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        'biosemi128': [
                                "C29", "C17", "C16",
                                "C30", "C19", "C8",
                    "D7", "D4", "C25", "C21", "C12", "C4", "C7",
                          "D8", "D10", "C23", "B29", "B27",
            "D23", "D21", "D19", "D14", "A1", "B20", "B22", "B24", "B26",
                          "D24", "D26", "A3", "B16", "B14",
                     "D31", "A7", "A5", "A19", "A32", "B4", "B11",
                          "A12", "A10", "A21", "B7", "B9",
                                 "A15", "A23", "A28",
                                        "A25",
        ]
    })

    valid_ratio: float = 0.20
    test_ratio: float = 0.0
    wnd_div_sec: int = 10
    suffix_path: str = 'BrainLat'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: ['AD', 'bvFTD', 'MS', 'HC',])


class BrainLatBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = BrainLatConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(
            name='finetune', is_finetune=True,
            valid_ratio=0.125, test_ratio=0.125,
        )
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()
        self.mapping_dict: dict[str, str] = {
            "C29": "Fp1",
            "C17": "Fpz",
            "C16": "Fp2",
            "C30": "AF7",
            "C19": "AFz",
            "C8": "AF8",

            "D7": "F7",
            "D4": "F3",
            "C25": "F1",
            "C21": "Fz",
            "C12": "F2",
            "C4": "F4",
            "C7": "F8",

            "D8": "FT7",
            "D10": "FC5",
            "C23": "FCz",
            "B29": "FC6",
            "B27": "FT8",

            "D23": "T7",
            "D21": "C5",
            "D19": "C3",
            "D14": "C1",
            "A1": "Cz",
            "B20": "C2",
            "B22": "C4",
            "B24": "C6",
            "B26": "T8",

            "D24": "TP7",
            "D26": "CP5",
            "A3": "CPz",
            "B16": "CP6",
            "B14": "TP8",

            "D31": "P7",
            "A7": "P3",
            "A5": "P1",
            "A19": "Pz",
            "A32": "P2",
            "B4": "P4",
            "B11": "P8",

            "A12": "PO9",
            "A10": "PO7",
            "A21": "POz",
            "B7": "PO8",
            "B9": "PO10",

            "A15": "O1",
            "A23": "Oz",
            "A28": "O2",

            "A25": "Iz",
        }


    def _load_meta_info(self):
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        dfs = []

        # !IMPORTANT csv data files from synapse is wrong for Demographics_PD_EEG_data.csv,
        # Cognition_PD_EEG_data.csv and Records_PD_EEG_data.csv for Google Drive changing the url.
        # You need to remove certain suffix after hash value in address to regain access to them.
        idx = [1, 2, 4, 5]
        for i, c in zip(idx, self.config.category):
            category_path = os.path.join(scan_path, f'{i}_{c}', f'Demographics_{c}_EEG_data.csv')
            df = pd.read_csv(category_path, sep=',')

            if c == 'MS':
                df = df.rename(columns={'id': 'path'})
                df['id EEG'] = df['id EEG'].str.replace(r'sub_', 'suj_', regex=True)
            df = df.drop(columns=['laterality'])
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].str.strip()

            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        self.sub_meta: DataFrame = df

    def _walk_raw_data_files(self):
        exclude_list = [
            'sub-100013', 'sub-100019',
            'sub-100023', 'sub-100025', 'sub-100027',
            'sub-100032', 'sub-100036', 'sub-100039',
            'sub-100040', 'sub-100041', 'sub-100042',
            'sub-100044', 'sub-100045', 'sub-100046',]

        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []

        for root, dirs, files in os.walk(scan_path):
            if '3_PD' in root:
                continue

            for file in files:
                if file.endswith(self.config.file_ext):

                    # exclude wrong files
                    skip_flag = False
                    if '5_HC' in root:
                        for ex in exclude_list:
                            if ex in root:
                                skip_flag = True

                    if skip_flag:
                        continue

                    file_path = os.path.join(root, file)
                    raw_data_files.append(os.path.normpath(file_path))

        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_dir = os.path.dirname(file_path)
        sub_pattern = r'/(suj_\d+|sub-\d+)'

        match = re.search(sub_pattern, file_dir)
        if match:
            subject = match.group(0)
        else:
            raise ValueError(f'Could not resolve file name: {file_dir}')

        group_pattern = r'/([1-5]_[A-Za-z]+)'
        match = re.search(group_pattern, file_dir)
        if match:
            group = match.group(0)
        else:
            raise ValueError(f'Could not resolve file name: {file_dir}')

        return {
            'subject': subject[1:],
            'session': 1,
            'group': group[3:],
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)

        montage = 'biosemi128'
        res = self.sub_meta[self.sub_meta['id EEG'] == info['subject']]
        if len(res) < 1:
            sex = None
            age = None
        else:
            res = res.iloc[0]
            sex = res['sex'].item()
            age = res['Age'].item()

        with self._read_raw_data(file_path) as raw:
            time = raw.duration

        info.update({
            'montage': montage,
            'time': time,
            'sex': sex,
            'age': age,
        })

        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        group = info['group']
        return [(group, 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        df = self._divide_all_split_by_sub_balanced(df)
        return df

    def _divide_all_split_by_sub_balanced(self, df: DataFrame):
        df['split'] = 'train'
        valid_ratio = self.config.valid_ratio
        test_ratio = self.config.test_ratio

        grouped = df.groupby('group')['subject'].unique()

        val_subjects = []
        test_subjects = []

        for group, subjects in grouped.items():
            subjects = list(subjects)
            np.random.shuffle(subjects)

            n_total = len(subjects)
            n_val = int(valid_ratio * n_total)
            n_test = int(test_ratio * n_total)

            if n_val + n_test > n_total:
                n_test = n_total - n_val

            n_test = max(n_test, 0)

            group_val = subjects[:n_val]
            group_test = subjects[n_val:n_val + n_test]

            val_subjects.extend(group_val)
            test_subjects.extend(group_test)

        df.loc[df['subject'].isin(val_subjects), 'split'] = 'valid'
        df.loc[df['subject'].isin(test_subjects), 'split'] = 'test'

        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [self.mapping_dict[ch].upper() for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _resample_and_filter(self, data: BaseRaw):
        # raw dataset has been filtered
        orig_fs = data.info['sfreq']
        if orig_fs != self.config.fs:
            data = data.resample(sfreq=self.config.fs, verbose=False)
        return data

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_eeglab(file_path, preload=preload, verbose=verbose)
            return raw

if __name__ == "__main__":
    builder = BrainLatBuilder("pretrain")
    builder.clean_disk_cache()
    builder.preproc(n_proc=2)
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
