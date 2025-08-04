import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import numpy as np
import pandas as pd
import s3fs
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class MotorMoveImagineConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "This data set consists of over 1500 one- and two-minute EEG recordings, obtained from 109 volunteers. "
        "Subjects performed different motor/imagery tasks while 64-channel EEG were recorded using the BCI2000 "
        "system (https://www.bci2000.org). Each subject performed 14 experimental runs: two one-minute baseline "
        "runs (one with eyes open, one with eyes closed), and three two-minute runs of each of the four following "
        "tasks: opens and closes the fist, opens and closes either both fists or both feet in both in imaginary "
        "and reality. https://physionet.org/content/eegmmidb/1.0.0/")
    citation: Optional[str] = """\
    @ARTICLE{1300799,
    author={Schalk, G. and McFarland, D.J. and Hinterberger, T. and Birbaumer, N. and Wolpaw, J.R.},
    journal={IEEE Transactions on Biomedical Engineering}, 
    title={BCI2000: a general-purpose brain-computer interface (BCI) system}, 
    year={2004},
    volume={51},
    number={6},
    pages={1034-1043},
    keywords={Brain computer interfaces;Signal processing;Protocols;Laboratories;Communication system control;Control systems;Signal processing algorithms;Research and development;Biomedical signal processing;Biomedical engineering},
    doi={10.1109/TBME.2004.827072}}
    """

    filter_notch: float = 60.0

    dataset_name: Optional[str] = 'motor_mv_img'
    task_type: DatasetTaskType = DatasetTaskType.MOTOR_IMAGINARY
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_10': [
                                            'Fp1.', 'Fpz.', 'Fp2.',
                                    'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.',
                    'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..',
                    'Ft7.', 'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'Ft8.',
            'T9..', 'T7..', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'T8..', 'T10.',
                    'Tp7.', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Tp8.',
                    'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
                                    'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                                            'O1..', 'Oz..', 'O2..',
                                                    'Iz..'
        ]
    })

    valid_ratio: float = 0.055
    test_ratio: float = 0.055
    wnd_div_sec: int = 12
    suffix_path: str = os.path.join('Motor Movement Imagery', 'eeg-motor-movementimagery-dataset-1.0.0')
    scan_sub_dir: str = "files"

    category: list[str] = field(default_factory=lambda: [
        'left', 'right', 'both_fist', 'foot'
    ])

class MotorMoveImagineBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = MotorMoveImagineConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True, wnd_div_sec=4)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self.task_ids = [4, 6, 8, 10, 12, 14]

    def _walk_raw_data_files(self):
        # noinspection PyTypeChecker
        scan_path: str = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []
        for root, dirs, files in os.walk(scan_path):
            for file in files:
                if file.endswith(self.config.file_ext):
                    subject, session = self._extract_file_name(file).split('R')
                    if int(session) in self.task_ids:
                        file_path = os.path.join(root, file)
                        raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, session = file_name.split('R')
        subject = subject[1:]
        return {
            'subject': int(subject),
            'session': int(session),
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': '10_10',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        session = info['session']
        if session in [4, 8, 12]:
            corr = {
                'T1': 'left',
                'T2': 'right',}
        else:
            corr = {
                'T1': 'both_fist',
                'T2': 'foot',}

        annotations = []
        with self._read_raw_data(file_path, preload=False, verbose=False) as data:
            assert data.annotations.onset[-1] < data.duration, "Invalid annotations"
            for i in range(len(data.annotations)):
                if data.annotations.description[i] in corr:
                    annotations.append((
                        corr[data.annotations.description[i]],
                        round(data.annotations.onset[i] * 1000),
                        round((data.annotations.onset[i] + self.config.wnd_div_sec) * 1000),
                    ))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        if self.config.is_finetune:
            df.loc[df['subject'].isin(np.arange(70)), 'split'] = 'train'
            df.loc[df['subject'].isin(np.arange(70, 89)), 'split'] = 'valid'
            df.loc[df['subject'].isin(np.arange(89, 110)), 'split'] = 'test'
        else:
            df.loc[df['subject'].isin(np.arange(89)), 'split'] = 'train'
            df.loc[df['subject'].isin(np.arange(89, 110)), 'split'] = 'valid'

        # df = self._divide_all_split_by_sub(df)
        # if not self.config.is_finetune:
        #     df.loc[df['split'] == 'test', 'split'] = 'valid'
        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs_std = [s.replace('.', '') for s in self.config.montage[montage]]
        chs_std = [ch.upper() for ch in chs_std]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _persist_example_file(self, sample: dict):
        # pretrain datasets have no ground truth will be assigned a label item which indicates all signal array
        path, montage, label, split = (
            sample['path'], sample['montage'], json.loads(sample['label']), sample['split'])
        try:
            with self._read_raw_data(path, preload=True, verbose=False) as data:
                if len(data.info['bads']) > 0:
                    data.interpolate_bads()
                data.set_eeg_reference(ref_channels='average', verbose=False)
                data = self._select_data_channels(data, path, montage)
    
                data = self._resample_and_filter(data)
                raw = self._fetch_signal_ndarray(data)
                chs_idx = self._fetch_chs_index(montage)
    
                examples = self._generate_window_sample(raw, montage, chs_idx, label, self.config.persist_drop_last)
                if len(examples) < 1:
                    return None
    
                df = pd.DataFrame(data=examples)
                filename = f"{self._encode_path(path)}.parquet"
                output_path = self._build_output_dir(split, filename)
    
                if self.config.is_remote_fs:
                    fs = s3fs.S3FileSystem(**self.s3_conf)
                    with fs.open(output_path, 'wb') as f:
                        df.to_parquet(
                            f,
                            compression=self.config.mid_compress_algo,
                            engine='pyarrow',
                            index=False)
                    fs.invalidate_cache()
                else:
                    df.to_parquet(
                        output_path,
                        compression=self.config.mid_compress_algo,
                        engine='pyarrow',
                        index=False)
        except Exception as e:
            logger.error(f"Error persisting example file {path}: {str(e)}")
            return None

        mid_df = pd.DataFrame(data={
            'key': [filename],
            'split': [split],
            'cnt': [len(examples)],})
        return mid_df


if __name__ == "__main__":
    builder = MotorMoveImagineBuilder('finetune')
    builder.preproc()
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
