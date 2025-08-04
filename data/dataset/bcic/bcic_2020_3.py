import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne.io
import pandas as pd
import s3fs
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class BCIC2020ImagineSpeechConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "EEG of five-class imagined speech words/phrases were recorded. 70 trials per class (70× 5 = 350 "
        "trials) are released for training (60 trials per class) and validation (10 trials per class) purpose. "
        "Using the given validation set is not obligated. Validation for the training data can be performed "
        "not only by the given validation set but also with the competitors’ choice (example: N-fold cross validation over the whole data). "
        "The test data (10 trials per class) will be released later. The dataset was divided into epochs based on cue information (event codes).")
    citation: Optional[str] = "https://osf.io/pq7vb/"

    filter_notch: float = 50.0
    is_notched: bool = True

    dataset_name: Optional[str] = 'bcic_2020_3'
    task_type: DatasetTaskType = DatasetTaskType.LINGUAL

    # And runs are seperated by multiple NaN in signal data
    file_ext: str = 'fif'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                                'Fp1',     'Fp2',
                          'AF7','AF3',     'AF4','AF8',
                  'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
        'FT9','FT7','FC5','FC3','FC1',     'FC2','FC4','FC6','FT8','FT10',
                  'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
        'TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10',
                  'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
                    'PO9','PO7','PO3','POz','PO4','PO8','PO10',
                                 'O1','Oz','O2',
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 3
    suffix_path: str = os.path.join('BCIC_2020_3')
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: [
        'hello', 'help me', 'stop', 'thank you', 'yes'
    ])


class BCIC2020ImagineBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = BCIC2020ImagineSpeechConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain', **kwargs):
        super().__init__(config_name, **kwargs)

    def _walk_raw_data_files(self):
        # noinspection PyTypeChecker
        scan_path: str = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []
        for root, dirs, files in os.walk(scan_path):
            for file in files:
                if file.endswith(self.config.file_ext):
                    file_path = os.path.join(root, file)
                    raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject = int(file_name.split('_')[1][-2:])
        session = 1

        return {
            'subject': subject,
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': '10_20',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            events, event_id = mne.events_from_annotations(raw)
            sf = raw.info['sfreq']
            events[:, 0] = events[:, 0] / sf

        mapping = {}
        for k, v in event_id.items():
            mapping[str(v)] = int(k)

        annotations = []
        for ev in events:
            label = 'default'
            if self.config.is_finetune:
                c = ev[2]
                c = mapping[str(c)]
                label = self.config.category[c]
            t_start = ev[0].item() * 1000
            annotations.append((
                label,
                round(t_start),
                round(t_start + self.config.wnd_div_sec * 1000)
            ))

        return annotations

    def _persist_example_file(self, sample: dict):
        # pretrain datasets have no ground truth will be assigned a label item which indicates all signal array
        path, montage, label, split = (
            sample['path'], sample['montage'], json.loads(sample['label']), sample['split'])
        try:
            with self._read_raw_data(path, preload=True, verbose=False) as data:
                data = self._select_data_channels(data, path, montage)
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

    def _divide_split(self, df: DataFrame) -> DataFrame:

        if self.config.is_finetune:
            df.loc[df['path'].str.contains('Training set'), 'split'] = 'train'
            df.loc[df['path'].str.contains('Validation set'), 'split'] = 'valid'
            df.loc[df['path'].str.contains('Test set'), 'split'] = 'test'
        else:
            df.loc[:, 'split'] = 'train'
            df.loc[df['path'].str.contains('Test set'), 'split'] = 'valid'

        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.upper() for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_fif(file_path, preload=preload, verbose=verbose)
            return raw


if __name__ == "__main__":
    builder = BCIC2020ImagineBuilder('finetune')
    # builder.clean_disk_cache()
    builder.preproc()
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)


