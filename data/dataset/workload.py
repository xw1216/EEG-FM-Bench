import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import pandas as pd
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class WorkloadConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "The dataset contains EEG recordings from 36 healthy volunteers during mental serial subtraction "
        "and corresponding reference background EEGs. Recorded using a 23 - channel system following the "
        "International 10/20 scheme, it's formatted according to BIDS standard. Based on task performance, "
        "subjects are divided into two groups. The dataset is available on PhysiobankL, useful for studying "
        "brain dynamics during cognitive workload.")
    citation: Optional[str] = """\
    @Article{data4010014,
    AUTHOR = {Zyma, Igor and Tukaev, Sergii and Seleznov, Ivan and Kiyono, Ken and Popov, Anton and Chernykh, Mariia and Shpenkov, Oleksii},
    TITLE = {Electroencephalograms during Mental Arithmetic Task Performance},
    JOURNAL = {Data},
    VOLUME = {4},
    YEAR = {2019},
    NUMBER = {1},
    ARTICLE-NUMBER = {14},
    URL = {https://www.mdpi.com/2306-5729/4/1/14},
    ISSN = {2306-5729},
    ABSTRACT = {This work has been carried out to support the investigation of the electroencephalogram (EEG) Fourier power spectral, coherence, and detrended fluctuation characteristics during performance of mental tasks. To this aim, the presented dataset contains International 10/20 system EEG recordings from subjects under mental cognitive workload (performing mental serial subtraction) and the corresponding reference background EEGs. Based on the subtraction task performance (number of subtractions and accuracy of the result), the subjects were divided into good counters and bad counters (for whom the mental task required excessive efforts). The data was recorded from 36 healthy volunteers of matched age, all of whom are students of Educational and Scientific Centre “Institute of Biology and Medicine”, National Taras Shevchenko University of Kyiv (Ukraine); the recordings are available through Physiobank platform. The dataset can be used by the neuroscience research community studying brain dynamics during cognitive workload.},
    DOI = {10.3390/data4010014}
    }
    """

    filter_notch: float = 50.0
    is_notched: bool = True

    dataset_name: Optional[str] = 'workload'
    task_type: DatasetTaskType = DatasetTaskType.WORKLOAD
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
            'EEG Fp1',
            'EEG Fp2',
            'EEG F7',
            'EEG Fz',
            'EEG F3',
            'EEG F4',
            'EEG F8',
            'EEG T3',
            'EEG C3',
            'EEG Cz',
            'EEG C4',
            'EEG T4',
            'EEG T5',
            'EEG P3',
            'EEG Pz',
            'EEG P4',
            'EEG T6',
            'EEG O1',
            'EEG O2',
        ]
    })

    valid_ratio: float = 0.14
    test_ratio: float = 0.14
    wnd_div_sec: int = 10
    suffix_path: str = 'Workload EEGMAT'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: ['background', 'arithmetic'])


class WorkloadBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = WorkloadConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True, wnd_div_sec=4)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        try:
            self.sub_meta = pd.read_csv(os.path.join(self.config.raw_path, 'subject-info.csv'))
        except FileNotFoundError as e:
            logger.error(e)
            raise e

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, session = file_name.split('_')
        return {
            'subject': subject,
            'session': int(session),
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        sex, age = self.sub_meta.loc[self.sub_meta['Subject'] == info['subject'], ['Gender', 'Age']]
        sex = 1 if sex == 'M' else 2
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': '10_20',
            'time': time,
            'sex': sex,
            'age': age,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        label = info['session'] - 1
        annotations = [(self.config.category[label], 0, -1)]
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.split(sep=' ')[1].upper() for ch in chs]
        chs_std = [self.montage_10_20_replace_dict.get(ch, ch).upper() for ch in chs_std]
        self._std_chs_cache[montage] = chs_std
        return chs_std


if __name__ == "__main__":
    builder = WorkloadBuilder("finetune")
    # builder.clean_disk_cache()
    # builder.preproc()
    builder.download_and_prepare(num_proc=8)
    dataset = builder.as_dataset()
    print(dataset)
