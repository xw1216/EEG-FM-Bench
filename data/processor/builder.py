import hashlib
import json
import logging
import math
import os
import shutil
import warnings
from abc import ABC
from dataclasses import dataclass,  field
from typing import Optional, Union, Any


import datasets

import mne
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from datasets import BuilderConfig, utils, DownloadManager, StreamingDownloadManager, SplitGenerator
from datasets.data_files import DataFilesDict, DataFilesPatternsDict
from multiprocess.pool import Pool
from mne.io import BaseRaw
from numpy import ndarray
from omegaconf import OmegaConf
from pandas import DataFrame
from tqdm import tqdm

from common.log import setup_log
from common.path import CONF_ROOT, DATABASE_CACHE_ROOT, DATABASE_PROC_ROOT, DATABASE_RAW_ROOT, LOG_ROOT, PLATFORM
from common.type import DatasetTaskType
from common.utils import ElectrodeSet


logger = logging.getLogger('preproc')


@dataclass
class EEGConfig(BuilderConfig):
    # basic info
    name: str = "pretrain"
    seed: int = 42
    version: Optional[Union[utils.Version, str]] = utils.Version("0.0.0")
    data_dir: Optional[str] = None
    data_files: Optional[Union[DataFilesDict, DataFilesPatternsDict]] = None
    description: Optional[str] = None
    citation: Optional[str] = None

    # preproc conf
    filter_low: float = 0.1
    filter_high: float = 128.0
    filter_notch: float = 50.0
    fs: float = 256.0
    unit: str = "uV"

    # middle cache storage
    mid_batch_size: int = 1e3
    mid_storage_format: str = 'parquet'
    mid_compress_algo: str = 'zstd'
    mid_max_files_per_dir: int = 1e4
    writer_batch_size: int = 512
    s3_delete_worker: int = 4

    # default database root path
    database_raw_root: str = DATABASE_RAW_ROOT
    database_proc_root: str = DATABASE_PROC_ROOT
    database_cache_root: str = DATABASE_CACHE_ROOT
    log_root: str = os.path.join(LOG_ROOT, 'preproc')
    s3_conf_path: str = os.path.join(
        CONF_ROOT, 's3', 's3_local.yaml'
        if PLATFORM == 'local' else 's3_remote.yaml')

    # auto defined
    raw_path: str = field(init=False)
    data_path: str = field(init=False)
    mid_path: str = field(init=False)
    wnd_len: int = field(init=False)
    is_remote_fs: bool = False

    # --- Subclass Override Conf ---
    # source dataset info
    dataset_name: Optional[str] = 'default'
    task_type: DatasetTaskType = DatasetTaskType.UNKNOWN
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {'default': []})

    # dynamic proc conf
    # preprocessed flag
    is_notched: bool = False
    persist_drop_last: bool = True

    # ratio may be useless if a raw dataset offers eval split
    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    # sample length
    wnd_div_sec: int = 10

    # dataset path conf
    suffix_path: str = ''
    scan_sub_dir: str = ''

    # finetune conf
    is_finetune: bool = False
    category: list[str] = field(default_factory=lambda: [])


    def __post_init__(self):
        super().__post_init__()
        self.raw_path = os.path.join(self.database_raw_root, self.suffix_path)
        self.data_path = os.path.join(self.database_proc_root)
        self.mid_path = os.path.join(self.database_cache_root, self.dataset_name)

        self.wnd_len = int(self.fs) * self.wnd_div_sec
        self.category_query_dict: dict[str, int] = {name: idx for idx, name in enumerate(self.category)}

        if not self.is_finetune:
            self.test_ratio = 0.0

        if self.dataset_name:
            self.log_root = os.path.join(self.log_root, self.dataset_name)

        if self.database_cache_root.startswith('s3://'):
            self.is_remote_fs = True

class EEGDatasetBuilder(datasets.GeneratorBasedBuilder, ABC):
    DEFAULT_CONFIG_NAME = 'pretrain'
    BUILDER_CONFIG_CLASS = EEGConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True),]

    def __init__(self, config_name='pretrain', **kwargs):
        conf: EEGConfig = self.builder_configs.get(config_name)
        super().__init__(
            cache_dir=conf.data_path,
            dataset_name=conf.dataset_name,
            config_name=config_name,
            writer_batch_size=conf.writer_batch_size,
            **kwargs
        )
        self.split_corr: dict = {
            'train': 0,
            'valid': 1,
            'test': 2
        }

        self.montage_10_20_replace_dict = {
            'T3': 'T7',
            'T4': 'T8',
            'T5': 'P7',
            'T6': 'P8'
        }

        self.electrode_set = ElectrodeSet()
        self.rng = np.random.default_rng(seed=self.config.seed)

        self.log_path = os.path.join(conf.log_root, f'{self.config.name}.log')
        self.log_err_files_path = os.path.join(conf.log_root, f'{self.config.name}_err_files.txt')
        setup_log(file_path=self.log_path, name='preproc')

        self.summary_path = os.path.join(conf.raw_path, 'summary', self.config.name)
        self.info_csv_path = os.path.join(self.summary_path, f'{self.dataset_name}_{self.config.name}_info.csv')
        self.mid_file_csv_path = os.path.join(self.summary_path, f'{self.dataset_name}_{self.config.name}_cache_files.csv')

        self._std_chs_cache:dict[str, list[str]] = {}
        self._std_chs_idx_cache: dict[str, list[int]] = {}

        if self.config.is_remote_fs:
            self.s3_conf = OmegaConf.load(self.config.s3_conf_path)
            self.s3_conf = OmegaConf.to_container(self.s3_conf, resolve=True)

    # GeneratorBasedBuilder Methods
    def _info(self) -> datasets.DatasetInfo:
        feat_dict = {
            "data": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
            "chs": datasets.Sequence(datasets.Value("int32")),
            "task": datasets.Value("int32"),
            "montage": datasets.Value("string"),
        }

        if self.config.is_finetune:
            # feat_dict.update({
            #     "label": datasets.ClassLabel(num_classes=len(self.config.category), names=self.config.category),
            # })
            feat_dict.update({
                "label": datasets.Value(dtype='int64')
            })

        features = datasets.Features(feat_dict)
        return datasets.DatasetInfo(
            dataset_name=self.config.dataset_name,
            config_name=self.config.name,
            description=self.config.description,
            citation=self.config.citation,
            features=features,
            version=self.config.version,
        )

    @staticmethod
    def select_split_to_dict(df: DataFrame, split: str):
        split_df: DataFrame = df[df['split'] == split]
        split_df.reset_index(drop=True, inplace=True)
        res_dict: dict[str, list] = split_df.to_dict(orient='list')
        return res_dict

    def _split_generators(self, dl_manager: Union[DownloadManager, StreamingDownloadManager]):
        # num_proc specified at download_and_prepare can split gen_kwargs into num_proc partition
        info_df = pd.read_csv(self.mid_file_csv_path)
        gen_list = [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=self.select_split_to_dict(info_df, 'train')),
            SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=self.select_split_to_dict(info_df, 'valid')),
        ]

        test_data = self.select_split_to_dict(info_df, 'test')
        if len(test_data) > 0 and self.config.is_finetune:
            gen_list.append(
                SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs=test_data),
            )
        return gen_list

    def _generate_examples(self, **kwargs):
        try:
            keys: list[str] = kwargs['key']
            splits: list[str] = kwargs['split']
            fs = s3fs.S3FileSystem(**self.s3_conf) if self.config.is_remote_fs else None
            for file, split in zip(keys, splits):
                file_path = os.path.join(self.config.mid_path, self.config.name, split, file)
                with (
                        fs.open(file_path, 'rb') if fs
                        else open(file_path, 'rb')  # 本地回退
                ) as f:
                    # disable internal multithread
                    table = pq.read_table(f, use_threads=False)
                    for idx in range(table.num_rows):
                        row = table.slice(idx, 1).to_pylist()[0]

                        row['chs'] = np.array(row['chs'], dtype=np.int32)
                        row['data'] = np.array(row['data'], dtype=np.float32).reshape(len(row['chs']), -1)
                        key = file + f'_{idx}'
                        yield key, row
        except Exception as e:
            logger.error(f"Error generating examples: {str(e)}")
            raise e

    def preproc(self, n_proc: Optional[int] = None):
        if self._is_preproc_cached():
            logger.info(f'Using cached summary info at {self.info_csv_path}')
            return

        if self.config.is_remote_fs:
            self._run_func_parallel(self._s3_link_test, [None], desc='Testing S3')

        np.random.seed(self.config.seed)
        self.clean_disk_cache()
        self.create_dir_structure()

        data_files = self._walk_raw_data_files()
        info_df = self._gather_data_info(data_files, n_proc)
        info_df = self._exclude_wrong_data(info_df, n_proc)
        split_df = self._divide_split(info_df)
        split_df.to_csv(self.info_csv_path, index=False)

        # split_df = pd.read_csv(self.info_csv_path)
        self._generate_middle_files(split_df, n_proc)

        self._mark_preproc_done()

    # Custom Methods
    def create_dir_structure(self):
        os.makedirs(self.summary_path, exist_ok=True)
        if self.config.is_remote_fs:
            return
        os.makedirs(self.config.mid_path, exist_ok=True)
        for split in ['train', 'valid', 'test'] if self.config.is_finetune else ['train', 'valid']:
            os.makedirs(os.path.join(self.config.mid_path, self.config.name, split), exist_ok=True)

    # noinspection PyUnusedLocal
    def _s3_link_test(self, data):
        try:
            logger.info(self.s3_conf)
            fs = s3fs.S3FileSystem(**self.s3_conf)
            if fs.exists(self.config.database_cache_root):
                logger.info('Remote cache dir exists')
            else:
                raise FileNotFoundError(f"Remote cache dir does not exist: {self.config.database_cache_root}")
        except Exception as e:
            logger.error(f"Can not resolve remote storage: {e}")
            raise e

    def _rm_s3_worker(self, path: str):
        fs = s3fs.S3FileSystem(**self.s3_conf)
        try:
            if fs.isdir(path):
                fs.rm(path, recursive=True)
            elif fs.isfile(path):
                fs.rm(path)
        except Exception as e:
            logger.warning(f"Warning: Failed to delete {path}, {str(e)}")

    def _list_s3_path(self, path: str):
        fs = s3fs.S3FileSystem(**self.s3_conf)
        try:
            return fs.glob(f'{path}/**/*.parquet')
        except FileNotFoundError:
            return []
        finally:
            fs.invalidate_cache()

    def _rm_s3_path(self, path: str, n_proc: Optional[int] = None):
        # !Important:
        # Do not use s3fs in multiprocess and default main process in the meantime,
        # or the latter execution for S3FileSystem will be stuck infinitely.
        # Please refer to https://s3fs.readthedocs.io/en/latest/#multiprocessing
        n_proc = n_proc if n_proc is not None else 1
        sub_paths = self._run_func_parallel(self._list_s3_path, [path], n_proc=1, desc=f'Listing S3 dir {path}')[0]
        self._run_func_parallel(self._rm_s3_worker, sub_paths, n_proc=n_proc, desc='Deleting S3 files')
        self._run_func_parallel(self._rm_s3_worker, [path], n_proc=1)

    def clean_arrow_set(self):
        try:
            if not self.config.data_path.startswith('s3://'):
                shutil.rmtree(os.path.join(self.config.data_path, self.config.dataset_name, self.config.name), ignore_errors=True)
            else:
                pass
                # self._rm_s3_path(os.path.join(self.config.data_path, self.config.dataset_name, self.config.name), n_proc=self.config.s3_delete_worker)
            logger.info(f'{self.config.dataset_name} arrow set cleared.')
        except Exception as e:
            logger.error(f'Error occurred during clean arrow dataset: {e}')

    def clean_disk_cache(self):
        try:
            shutil.rmtree(self.summary_path, ignore_errors=True)
            if not self.config.is_remote_fs:
                shutil.rmtree(os.path.join(self.config.mid_path, self.config.name), ignore_errors=True)
            else:
                pass
                # self._rm_s3_path(self.config.mid_path, n_proc=self.config.s3_delete_worker)
            logger.info(f'{self.config.dataset_name} cache cleared.')
        except FileNotFoundError as e:
            logger.error(f'{self.config.dataset_name} cache not exist: {e}')
        except PermissionError:
            logger.error(f'Permission Denied')
        except Exception as e:
            logger.error(f'Error occurred during clean builder cache: {e}')

    def _generate_middle_files(self, df: DataFrame, n_proc: Optional[int] = None):
        rows = df.to_dict(orient='records')
        results = self._run_func_parallel(
            self._persist_example_file, rows, n_proc=n_proc,
            desc='Generating wnd samples and persisting parquet files')

        mid_dfs = [item for item in results if item is not None]
        mid_df = pd.concat(mid_dfs, ignore_index=True, axis=0)
        mid_df.to_csv(self.mid_file_csv_path, index=False)

    def _build_output_dir(self, split: str, filename: str):
        base_path: str = self.config.mid_path
        if self.config.is_remote_fs:
            return f"{base_path.rstrip('/')}/{self.config.name}/{split}/{filename}"
        # logger.info(f'{base_path} {self.config.name} {split} {filename}')
        return os.path.join(base_path, self.config.name, split, filename)

    def _persist_example_file(self, sample: dict):
        # pretrain datasets have no ground truth will be assigned a label item which indicates all signal array
        path, montage, label, split = (
            sample['path'], sample['montage'], json.loads(sample['label']), sample['split'])
        try:
            with self._read_raw_data(path, preload=True, verbose=False) as data:
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

    def _generate_window_sample(
            self,
            raw: ndarray,
            montage: str,
            chs_idx: ndarray,
            labels:list[tuple[str, int, int]],
            drop_last: bool=True,
    ):
        """
        Generates windowed samples from raw EEG data using a sliding window approach. The function extracts windows
        of a fixed length from specified channels, adjusts window boundaries if necessary, and prepares them for
        further processing. It also allows for the inclusion of the last segment of the data if configured.

        :param raw: A matrix where rows represent EEG signal channels and columns represent time points.
        :param montage: The montage name representing the specific electrode placement setup for the EEG data.
        :param chs_idx: Array of indices specifying the channels selected for window generation.
        :param labels: A list of tuples, where each element has a string label, start time, and end time (in milliseconds).
        :param drop_last: A flag indicating whether to drop the last window if its length is less than the configured
            window length. Defaults to True.
        :return: A list of dictionaries, each containing windowed data and associated metadata (e.g., channel indices,
            montage, task type). If the data length is insufficient for even a single window, None is returned.
        """
        wnds = []

        signal_len = raw.shape[1]
        if signal_len < self.config.wnd_len:
            return wnds

        for label, start_t, end_t in labels:
            if self.config.is_finetune and label not in self.config.category:
                continue

            start = self._milli_sec_to_pts(start_t)
            end = signal_len if end_t < 0 else self._milli_sec_to_pts(end_t)
            if end > signal_len or start < 0 or start >= end:
                continue

            label_idx = self.config.category_query_dict[label] if self.config.is_finetune else 0
            n_wnd, remain_pts = divmod(end - start, self.config.wnd_len)

            positions = start + np.arange(n_wnd)[:, None] * self.config.wnd_len
            indices = positions + np.arange(self.config.wnd_len)
            # wnd_data_batch shape (n_channels, n_windows, window_length) -> (n_windows, n_channels, window_length)
            wnd_data_batch = raw[:, indices]
            # wnd_data_batch will be empty ndarray if n_wnd is 0
            wnd_data_batch: ndarray = np.transpose(wnd_data_batch, (1, 0, 2))

            base_dict = {
                'chs': chs_idx,
                'montage': f'{self.config.dataset_name}/{montage}',
                'task': self.config.task_type.value,
            }

            example_dicts = []
            for wnd_data in wnd_data_batch:
                assert wnd_data.shape[1] == self.config.wnd_len

                example_dict = base_dict.copy()
                example_dict['data'] = np.ascontiguousarray(wnd_data.flatten().astype(np.float32))
                if self.config.is_finetune:
                    example_dict['label'] = label_idx
                example_dicts.append(example_dict)

            if not drop_last and remain_pts > 0:
                if end - self.config.wnd_len >= 0:
                    pos = end - self.config.wnd_len
                elif start + self.config.wnd_len <= signal_len:
                    pos = start
                else:
                    offset = self.config.wnd_len - (signal_len - end + remain_pts)
                    pos = start - offset
                    assert pos < 0

                wnd_data = raw[:, pos: pos + self.config.wnd_len]
                assert wnd_data.shape[1] == self.config.wnd_len

                example_dict = base_dict.copy()
                example_dict['data'] = np.ascontiguousarray(wnd_data.flatten().astype(np.float32))
                if self.config.is_finetune:
                    example_dict['label'] = label_idx
                example_dicts.append(example_dict)
            wnds.extend(example_dicts)
        return wnds

    def _mark_preproc_done(self):
        with open(os.path.join(self.summary_path, f'{self.config.name}.done'), 'w'):
            pass

    def _is_preproc_cached(self):
        return os.path.exists(os.path.join(self.summary_path, f'{self.config.name}.done'))

    def _walk_raw_data_files(self):
        logger.info('Walking eeg data files...')
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []
        for root, dirs, files in os.walk(scan_path):
            for file in files:
                if file.endswith(self.config.file_ext):
                    file_path = os.path.join(root, file)
                    raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _gather_data_info(self, data_files: list[str], n_proc: Optional[int] = None):
        results = self._run_func_parallel(
            self._gather_files,
            data_files,
            n_proc=n_proc,
            desc='Gathering metadata'
        )
        data_info = []
        for result in results:
            if result is not None:
                data_info.append(result)
        df = DataFrame(data_info)
        return df

    def _gather_files(self, data: str):
        try:
            info = {'path': data}
            info.update(self._resolve_exp_meta_info(data))
            annotations = self._resolve_exp_events(data, info)
            info.update({'label': json.dumps(annotations)})
            return info
        except Exception as e:
            logger.error(f"Error accessing metadata in file {data}: {str(e)}")
            return None

    def _check_data_length(self, df: DataFrame):
        mask = df['time'] >= float(self.config.wnd_div_sec)
        filtered_df = df[mask].reset_index(drop=True)
        return filtered_df
        # indices_to_drop = df.loc[df['time'] < float(self.config.wnd_div_sec)].index
        # df.drop(indices_to_drop, inplace=True)
        # df.reset_index(drop=True, inplace=True)
        # return df

    def _exclude_wrong_data(self, df: DataFrame, n_proc: Optional[int] = None):
        df = self._check_data_montage_channel(df, n_proc=n_proc)
        df = self._check_data_length(df)
        return df

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        """
        Parse all info from file name and return a dict.
        :param file_path: absolute path of eeg raw data file
        :return: info dict
        """
        raise NotImplementedError

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        """
        Meta info must have ``subject,session,montage,time``.
        If dataset has been split into partitions, ``split`` must be included.
        Time is recording seconds in float.

        :param file_path: absolute path of eeg raw data file
        :return: info dict
        """
        raise NotImplementedError

    def _resolve_exp_events(self, file_path: str, info:dict[str, Any]):
        """
        If data is labeled in a whole,
        a label column contains a str must be included.

        Otherwise, if data is labeled by annotations,
        a label column contains a list which records (label, start, end) timing
        windows tuples in millisecond of all type of annotations must be included.

        :param file_path:
        :return: dict[str, list[tuple[str, int, int]]]
        """
        raise NotImplementedError

    def _divide_split(self, df: DataFrame) -> DataFrame:
        """
        A ``split`` column will be added to rows in dataframe
        which stands for a single data file.

        Split will be one of ``['train','valid','test']`` and assigned by grouping subject

        Dataframe will be skipped if ``split`` column exists.
        :param df:
        :return: DataFrame
        """
        raise NotImplementedError

    def _get_chs_name_by_montage(self, montage_name: str, is_std: bool=False):
        if is_std:
            return self.standardize_chs_names(montage_name)
        else:
            return self.config.montage[montage_name]

    def standardize_chs_names(self, montage: str):
        raise NotImplementedError

    def _read_raw_data(self, file_path: str, preload: bool=False, verbose: bool=False) -> BaseRaw:
        if self.config.file_ext == 'edf':
            data = mne.io.read_raw_edf(file_path, preload=preload, verbose=verbose)
        elif self.config.file_ext == 'bdf':
            data = mne.io.read_raw_bdf(file_path, preload=preload, verbose=verbose)
        elif self.config.file_ext == 'cnt':
            data = mne.io.read_raw_cnt(file_path, preload=preload, verbose=verbose)
        elif self.config.file_ext == 'gdf':
            data = mne.io.read_raw_gdf(file_path, preload=preload, verbose=verbose)
        elif self.config.file_ext == 'fif':
            data = mne.io.read_raw_fif(file_path, preload=preload, verbose=verbose)
        elif self.config.file_ext == 'set':
            data = mne.io.read_raw_eeglab(file_path, preload=preload, verbose=verbose)
        else:
            raise NotImplementedError(f"Can't load raw eeg data in {self.config.file_ext} format.")
        return data

    def _check_montage_single_file(self, row: dict):
        file_path = row['path']
        montage = row['montage']
        with self._read_raw_data(file_path) as data:
            src_chs = set(data.ch_names)
            chs = set(self._get_chs_name_by_montage(montage))

        if src_chs.intersection(chs) != chs:
            logger.warning(f'Channel config is wrong for file: {file_path}. Loss channel: {chs.difference(src_chs)}.')
            return False
        return True

    def _check_data_montage_channel(self, df: DataFrame, n_proc: Optional[int] = None):
        """
        Checks the validity of the channel information in the input DataFrame based
        on the associated montage and file data. Identifies and removes raw data rows with
        inconsistent channel information, and logs paths of files with errors to a
        specified error log file.

        :param df: A pandas DataFrame containing channel and montage information.
            Each row should represent a file, with the column 'path' specifying the
            file path, and 'montage' specifying the montage configuration.
        :return: A pandas DataFrame with rows containing inconsistent channel
            information removed and indexed reset.
        """
        rows = df.to_dict(orient='records')
        results = self._run_func_parallel(
            self._check_montage_single_file,
            rows,
            n_proc=n_proc,
            desc='Checking montage channel'
        )
        sel = np.array(results, dtype=np.bool)

        wrong_files = df.loc[~sel, 'path'].tolist()
        with open(self.log_err_files_path, 'a') as f:
            for file_path in wrong_files:
                f.write(f'{file_path}\n')

        df = df[sel]
        df.reset_index(drop=True, inplace=True)
        return df

    def _select_data_channels(self, data: BaseRaw, file_path: str, montage: str):
        chs = self._get_chs_name_by_montage(montage)
        drop_chs = list((set(data.ch_names) - set(chs)))
        data = data.drop_channels(drop_chs)
        if len(chs) != len(data.ch_names):
            raise RuntimeError(f'Channel config is wrong for file: {file_path}')

        data.reorder_channels(chs)
        if data.ch_names != chs:
            raise RuntimeError(f'Failed to reorder desired channels: {file_path}')
        return data

    def _resample_and_filter(self, data: BaseRaw):
        orig_fs = data.info['sfreq']
        # mne lowpass and high pass in raw info are unreliable
        filter_param = {
            'verbose': False,
            'h_freq': self.config.filter_high if orig_fs > self.config.filter_high * 2 else None,
            'l_freq': self.config.filter_low
        }

        filter_length = round(3.3 * (1 / min(max(filter_param['l_freq'] * 0.25, 2), filter_param['l_freq'])) * orig_fs)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings(
                "always",
                category=RuntimeWarning,
            )
            if data.duration * orig_fs > filter_length:
                data = data.filter(**filter_param)
            for warn in w:
                raise warn

        if not self.config.is_notched:
            notch_freqs = np.arange(self.config.filter_notch, orig_fs / 2, self.config.filter_notch).tolist()
            data = data.notch_filter(freqs=notch_freqs, verbose=False)
        if orig_fs != self.config.fs:
            data = data.resample(sfreq=self.config.fs, verbose=False)
        return data

    def _fetch_signal_ndarray(self, data: BaseRaw) -> ndarray:
        return data.get_data(units=self.config.unit).astype(np.float32).copy()

    def _fetch_chs_index(self, montage: str):
        if montage in self._std_chs_idx_cache.keys():
            return self._std_chs_idx_cache[montage].copy()
        chs = self._get_chs_name_by_montage(montage, is_std=True)
        idx = np.ascontiguousarray(self.electrode_set.get_electrodes_index(chs))
        self._std_chs_idx_cache[montage] = idx
        return idx.copy()

    def _milli_sec_to_pts(self, time: int):
        return math.floor(time * self.config.fs / 1000)

    def _divide_all_split_by_sub(self, df: DataFrame):
        df['split'] = 'train'
        train_subjects = df.loc[:, 'subject'].unique()
        n_val_sub = int(len(train_subjects) * self.config.valid_ratio)
        n_test_sub = int(len(train_subjects) * self.config.test_ratio)

        selection = np.random.choice(train_subjects, n_val_sub + n_test_sub, replace=False)
        val_subjects = selection[:n_val_sub]
        test_subjects = selection[n_val_sub:]

        df.loc[df['subject'].isin(val_subjects), 'split'] = 'valid'
        df.loc[df['subject'].isin(test_subjects), 'split'] = 'test'
        return df

    def _divide_test_from_valid_by_sub(self, df: DataFrame):
        if not self.config.is_finetune:
            return df
        valid_subjects = df.loc[df['split'] == 'valid', 'subject'].unique()
        n_val_sub = math.ceil(len(valid_subjects) / 2)

        selection = np.random.choice(valid_subjects, n_val_sub, replace=False)
        val_subjects = selection[:n_val_sub]
        test_subjects = selection[n_val_sub:]

        df.loc[df['subject'].isin(val_subjects), 'split'] = 'valid'
        df.loc[df['subject'].isin(test_subjects), 'split'] = 'test'
        return df

    @staticmethod
    def _iterative_greedy_split(y_weighted, ratios):
        n_subjects = y_weighted.shape[0]
        total_weights = y_weighted.sum(axis=0)
        ratios = np.array(ratios)
        ratios = ratios * (1 / ratios.sum())

        splits = [{
            'indices': [],
            'current_weights': np.zeros(y_weighted.shape[1]),
            'target_weights': ratio * total_weights
        } for ratio in ratios]

        remaining_indices = list(range(n_subjects))
        label_order = np.argsort(total_weights)
        for l in label_order:
            related = [(i, y_weighted[i, l]) for i in remaining_indices
                       if y_weighted[i, l] > 0]
            related.sort(key=lambda x: -x[1])

            for i, _ in related:
                best_split = None
                min_def = float('inf')

                for s in splits:
                    deficit = s['target_weights'][l] - s['current_weights'][l]
                    if deficit <= 0:
                        continue

                    after_add = s['current_weights'][l] + y_weighted[i, l]
                    new_def = abs(s['target_weights'][l] - after_add)

                    if new_def < min_def:
                        min_def = new_def
                        best_split = s

                if best_split:
                    best_split['indices'].append(i)
                    best_split['current_weights'] += y_weighted[i]
                    remaining_indices.remove(i)

        for i in remaining_indices:
            closest = np.argmin([
                np.linalg.norm(s['current_weights'] + y_weighted[i] - s['target_weights'])
                for s in splits
            ])
            splits[closest]['indices'].append(i)
            splits[closest]['current_weights'] += y_weighted[i]

        return [np.array(s['indices']) for s in splits]

    def _analyze_split(self, y_weighted, split_indices, splits_name: list[str]):
        assert  len(split_indices) == len(splits_name)
        total = y_weighted.sum(axis=0)
        split_ratios = [y_weighted[split, :].sum(axis=0) for split in split_indices]

        logger.info('SPLIT RESULT')
        info = f"{'LABEL':<15} | {'TOTAL':<10}"
        for split in splits_name:
            info = info + f" | {f'{split}%':<8}"
        logger.info(info)
        for i in range(len(self.config.category)):
            info = f"{self.config.category[i]:<15} | {total[i]:<10.0f}"
            for j in range(len(splits_name)):
                info = info + f" | {split_ratios[j][i] / total[i]:<8.2%}"
            logger.info(info)

        dist = np.array(split_ratios).sum(axis=1) / total.sum()
        info = f"{'SUM':<15} | {f'{total.sum()}':<10}"
        for j in range(len(splits_name)):
            info = info + f" | {dist[j]:<8.2%}"
        logger.info(info)

    def _multi_label_iterative_stratified_split(self, df: DataFrame, splits_name: list[str]) -> DataFrame:
        if not self.config.is_finetune:
            logger.warning('Multi-label iterative stratified split should only be available for finetune')
        # simplifies x, y structure
        subjects = df['subject'].tolist()
        labels = df['label'].tolist()
        times = df['time'].tolist()
        for i in range(len(labels)):
            labels[i] = json.loads(labels[i])
            labels_new, labels_wnd = [], []
            for label in labels[i][:]:
                if label[0] not in self.config.category:
                    continue
                else:
                    idx = self.config.category.index(label[0])
                    start = self._milli_sec_to_pts(label[1])
                    end = self._milli_sec_to_pts(label[2] if label[2] > 0 else times[i] * 1000)
                    n_wnd, remain_pts = divmod(end - start, self.config.wnd_len)
                    if not self.config.persist_drop_last and remain_pts > 0:
                        n_wnd += 1

                    labels_new.append(idx)
                    labels_wnd.append(n_wnd)
            labels[i] = (labels_new, labels_wnd)

        unique_subjects = np.unique(subjects)
        label_names = self.config.category
        y_weighted = np.zeros((len(unique_subjects), len(label_names)), dtype=np.int64)
        subject_idx = {s: i for i, s in enumerate(unique_subjects)}

        for subj, label_tuple in zip(subjects, labels):
            idx = subject_idx[subj]
            label, wnd = label_tuple
            weight = np.bincount(np.array(label), weights=np.array(wnd, dtype=np.int64), minlength=len(self.config.category))
            y_weighted[idx] += weight.astype(np.int64)

        ratios = np.array([1 - self.config.valid_ratio - self.config.test_ratio,
                           self.config.valid_ratio, self.config.test_ratio], dtype=np.float32)
        split_mask = np.array([split in splits_name for split in self.split_corr], dtype=bool)
        split_indices = self._iterative_greedy_split(y_weighted, ratios=ratios[split_mask])

        self._analyze_split(y_weighted, split_indices, splits_name)

        for split, indices in zip(splits_name, split_indices):
            df.loc[df['subject'].isin(unique_subjects[indices]), 'split'] = split
        return df

    def _divide_label_balance_all_split(
            self,
            df: DataFrame,
            splits_name: list[str] = None,
    ) -> DataFrame:
        if splits_name is None:
            splits_name = ['train', 'valid', 'test']

        return self._multi_label_iterative_stratified_split(df, splits_name=splits_name)

    def _divide_balance_test_from_valid(self, df: DataFrame):
        # no test subject should be in valid set
        if not self.config.is_finetune:
            return df

        train_df = df.loc[df['split'] == 'train', :]
        valid_df = df.loc[df['split'] == 'valid', :]

        valid_df = self._multi_label_iterative_stratified_split(valid_df, splits_name=['valid', 'test'])

        df = pd.concat([train_df, valid_df], axis=0, ignore_index=True, sort=False)
        df.reset_index(drop=True, inplace=True)
        return df

    def _convert_to_mne(self, data, info) -> mne.io.RawArray:
        """
        Converts the provided `data` and `info` parameters into an MNE-compatible
        format. This method is intended to be implemented by subclasses to
        specify their particular approach for converting the input data.

        :param data: The input data to be converted. The structure and type of
            this data depend on the implementation and may vary across subclasses.
        :param info: The metadata or information associated with the input `data.`
            It provides context or descriptive details that will be used in the
            conversion process.
        :return: The converted MNE-compatible data. Exact type and structure
            depend on the specific subclass implementation.
        """
        raise NotImplementedError

    @staticmethod
    def _merge_overlap_labels(group):
        """
        Merges overlapping time intervals within a given group of time labels. The function
        assumes that each group contains time intervals defined by `start_time` and `stop_time`.
        If the `start_time` of any interval overlaps or touches the `stop_time` of another, they
        are merged into a single interval with `stop_time` as the maximum of the overlapping
        intervals' `stop_time`.

        :param group: A pandas DataFrame containing the time intervals to be merged. It is
            assumed to have the columns `start_time` and `stop_time`. Additionally, it
            assumes the group is associated with a label indicated by `group.name`.
        :type group: pandas.DataFrame
        :return: A new pandas DataFrame containing the merged time intervals. Each row
            consists of a `start_time`, `stop_time`, and the associated group `label`,
            which corresponds to the original group name.
        :rtype: pandas.DataFrame
        """
        sorted_group = group.sort_values('start_time')
        merged = []

        for _, row in sorted_group.iterrows():
            if not merged:
                merged.append({'start_time': row['start_time'], 'stop_time': row['stop_time']})
            else:
                last = merged[-1]
                if row['start_time'] <= last['stop_time']:
                    last['stop_time'] = max(last['stop_time'], row['stop_time'])
                else:
                    merged.append({'start_time': row['start_time'], 'stop_time': row['stop_time']})

        merged_df = pd.DataFrame(merged)
        merged_df['label'] = group.name
        return merged_df

    @staticmethod
    def _run_func_parallel(
            func,
            data: list,
            chunk_size: Optional[int]=None,
            n_proc: Optional[int]=None,
            desc: str= 'Processing'
    ):
        if n_proc is None:
            # n_proc = max(1, round(get_available_cpu() / 2))
            n_proc = 1
        if chunk_size is None:
            chunk_size = 1
        logger.info(f"Run {func.__name__} parallel in {n_proc} processes with chunksize {chunk_size}")

        results = []
        with Pool(n_proc) as pool:
            with tqdm(total=len(data), desc=desc) as pbar:
                for res in pool.imap(func, data, chunksize=chunk_size):
                    results.append(res)
                    pbar.update(1)

        return results

    @staticmethod
    def _extract_file_name(file_path: str):
        return os.path.basename(file_path).split('.')[0]

    @staticmethod
    def _extract_middle_path(file_path: str, s: int, e: int):
        return file_path.split('/')[s:e]

    @staticmethod
    def _encode_path(file_path: str):
        return hashlib.sha512(file_path.encode()).hexdigest()

if __name__ == "__main__":
    pass
