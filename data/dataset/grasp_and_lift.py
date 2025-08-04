import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne.io
import numpy as np
import pandas as pd
from mne.io import BaseRaw
from numpy import ndarray
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class GraspAndLiftConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "There are 12 subjects in total, 10 series of trials for each subject, and approximately "
        "30 trials within each series. The number of trials varies for each series. The training "
        "set contains the first 8 series for each subject. The test set contains the 9th and 10th series. "
        "For each GAL, you are tasked to detect 6 events: HandStart FirstDigitTouch BothStartLoadPhase "
        "LiftOff Replace BothReleased These events always occur in the same order. In the training set, "
        "there are two files for each subject + series combination: the *_data.csv files contain the raw "
        "32 channels EEG data (sampling rate 500Hz) the *_events.csv files contains the ground truth "
        "frame-wise labels for all events")
    citation: Optional[str] = """\
    @Article{Luciw2014,
    author={Luciw, Matthew D.
    and Jarocka, Ewa
    and Edin, Benoni B.},
    title={Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction},
    journal={Scientific Data},
    year={2014},
    month={Nov},
    day={25},
    volume={1},
    number={1},
    pages={140047},
    issn={2052-4463},
    doi={10.1038/sdata.2014.47},
    url={https://doi.org/10.1038/sdata.2014.47}}
    """

    filter_notch: float = 50.0
    orig_fs: float = 500.0

    dataset_name: Optional[str] = 'grasp_and_lift'
    task_type: DatasetTaskType = DatasetTaskType.MOTOR_EXECUTION
    file_ext: str = 'csv'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        'ActiCap32': [
                          'Fp1', 'Fp2',
                  'F7', 'F3', 'Fz', 'F4', 'F8',
                   'FC5', 'FC1', 'FC2', 'FC6',
                  'T7', 'C3', 'Cz', 'C4', 'T8',
            'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
                   'P7', 'P3','Pz', 'P4', 'P8',
                 'PO9', 'O1', 'Oz', 'O2', 'PO10',
        ]
    })

    valid_ratio: float = 0.167
    test_ratio: float = 0.0
    wnd_div_sec: int = 5
    suffix_path: str = os.path.join('Grasp and Lift EEG Challenge', 'grasp-and-lift-eeg-detection')
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: [])


class GraspAndLiftBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = GraspAndLiftConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _walk_raw_data_files(self):
        raw_data_files = super()._walk_raw_data_files()
        for file in raw_data_files[:]:
            if file.split('_')[-1] != 'data.csv':
                raw_data_files.remove(file)
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        subject, session, term = file_name.split('_')
        session = int(session[6:])
        return {
            'subject': subject,
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        data = pd.read_csv(file_path, sep=',', header='infer')

        time = len(data) / self.config.orig_fs

        info.update({
            'montage': 'ActiCap32',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        return [('default', 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs_std = [ch.upper() for ch in self.config.montage[montage]]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        data = pd.read_csv(file_path, sep=',', header='infer')
        return self._convert_to_mne(data, None)

    def _convert_to_mne(self, data: DataFrame, info) -> mne.io.RawArray:
        ch_names = self.config.montage['ActiCap32']
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.config.orig_fs,
            ch_types=['eeg'] * len(ch_names))

        signal_data: ndarray = data.iloc[:, 1:].to_numpy()
        signal_data = signal_data.astype(np.float32).transpose(1, 0)

        # mne expected unit is Volts, so turn 0.1 uV to V
        raw = mne.io.RawArray(signal_data / 1e7, info, verbose=False)
        return raw
    
if __name__ == "__main__":
    builder = GraspAndLiftBuilder('pretrain')
    builder.preproc()
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
