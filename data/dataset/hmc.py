import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import pandas as pd
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class HMCConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.1.0")
    description: Optional[str] = (
        "A collection of 151 whole-night polysomnographic (PSG) sleep recordings (85 Male, 66 Female, mean Age of 53.9 ± 15.4)"
        " collected during 2018 at the Haaglanden Medisch Centrum (HMC, The Netherlands) sleep center. "
        "Patient recordings were randomly selected and include a heterogeneous population which was referred for PSG "
        "examination on the context of different sleep disorders. The dataset contains electroencephalographic (EEG), "
        "electrooculographic (EOG), chin electromyographic (EMG), and electrocardiographic (ECG) activity, "
        "as well as event annotations corresponding to scoring of sleep patterns (hypnogram) performed by sleep technicians at HMC. "
        "The dataset was collected as part of a study evaluating the generalization performance of an automatic sleep scoring algorithm across multiple heterogeneous datasets.")
    citation: Optional[str] = """\
    @article{10.1371/journal.pone.0256111,
    doi = {10.1371/journal.pone.0256111},
    author = {Alvarez-Estevez, Diego AND Rijsman, Roselyne M.},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Inter-database validation of a deep learning approach for automatic sleep scoring},
    year = {2021},
    month = {08},
    volume = {16},
    url = {https://doi.org/10.1371/journal.pone.0256111},
    pages = {1-27},
    abstract = {Study objectives Development of inter-database generalizable sleep staging algorithms represents a challenge due to increased data variability across different datasets. Sharing data between different centers is also a problem due to potential restrictions due to patient privacy protection. In this work, we describe a new deep learning approach for automatic sleep staging, and address its generalization capabilities on a wide range of public sleep staging databases. We also examine the suitability of a novel approach that uses an ensemble of individual local models and evaluate its impact on the resulting inter-database generalization performance.   Methods A general deep learning network architecture for automatic sleep staging is presented. Different preprocessing and architectural variant options are tested. The resulting prediction capabilities are evaluated and compared on a heterogeneous collection of six public sleep staging datasets. Validation is carried out in the context of independent local and external dataset generalization scenarios.   Results Best results were achieved using the CNN_LSTM_5 neural network variant. Average prediction capabilities on independent local testing sets achieved 0.80 kappa score. When individual local models predict data from external datasets, average kappa score decreases to 0.54. Using the proposed ensemble-based approach, average kappa performance on the external dataset prediction scenario increases to 0.62. To our knowledge this is the largest study by the number of datasets so far on validating the generalization capabilities of an automatic sleep staging algorithm using external databases.   Conclusions Validation results show good general performance of our method, as compared with the expected levels of human agreement, as well as to state-of-the-art automatic sleep staging methods. The proposed ensemble-based approach enables flexible and scalable design, allowing dynamic integration of local models into the final ensemble, preserving data locality, and increasing generalization capabilities of the resulting system at the same time.},
    number = {8},}
    """

    filter_notch: float = 50.0

    dataset_name: Optional[str] = 'hmc'
    task_type: DatasetTaskType = DatasetTaskType.SLEEP_STAGE
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        'AASM_24_Minimal': [
            'EEG F4-M1',
            'EEG C4-M1',
            'EEG O2-M1',
            'EEG C3-M2'
        ]
    })

    valid_ratio: float = 0.162
    test_ratio: float = 0.162
    wnd_div_sec: int = 30
    suffix_path: str = "HMC"
    scan_sub_dir: str = "recordings"

    category: list[str] = field(default_factory=lambda: [
        'W', 'R', 'N1', 'N2', 'N3'
    ])


class HMCBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = HMCConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _walk_raw_data_files(self):
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        raw_data_files = []
        for root, dirs, files in os.walk(scan_path):
            for file in files:
                if file.endswith(self.config.file_ext) and len(self._extract_file_name(file).split('_')) == 1:
                    file_path = os.path.join(root, file)
                    raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject = file_name[2:]
        return {
            'subject': int(subject),
            'session': 1,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration

        info.update({
            'montage': 'AASM_24_Minimal',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        file_name = self._extract_file_name(file_path)
        event_file_name = file_name + '_sleepscoring.txt'
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir, event_file_name)
        df = pd.read_csv(scan_path, sep=r'\s*,\s*', header=0, engine='python')

        # Find valid sleeping duration, and exclude slices do not up to 30s
        start_idx = df.loc[df['Annotation'] == 'Lights off'].index.item() + 1
        end_idx = df.loc[df['Annotation'] == 'Lights on'].index.item()
        df = df.iloc[start_idx:end_idx, :]
        df = df.loc[df['Duration'] >= 30]

        annotations = []
        for i in range(len(df)):
            annotations.append((
                df.iloc[i, df.columns.get_loc('Annotation')].split(' ')[-1],
                round(df.iloc[i, df.columns.get_loc('Recording onset')] * 1000),
                round((df.iloc[i, df.columns.get_loc('Recording onset')] + 30) * 1000),
            ))
        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.split(sep=' ')[1].split('-')[0] for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_edf(file_path, preload=preload, verbose=verbose)
            return raw


if __name__ == "__main__":
    builder = HMCBuilder('finetune')
    # builder.clean_disk_cache()
    # builder.preproc()
    builder.download_and_prepare(num_proc=8)
    dataset = builder.as_dataset()
    print(dataset)
