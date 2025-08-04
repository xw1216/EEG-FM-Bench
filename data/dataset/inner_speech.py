import logging
import warnings
from dataclasses import dataclass, field
from math import floor
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
from mne.io import BaseRaw
from numpy import ndarray
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class InnerSpeechConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "In order to improve the understanding of inner speech and its applications in real BCIs systems, we have built "
        "a multi speech-related BCI dataset consisting of EEG recordings from ten naive BCI users, performing four "
        "mental tasks in three different conditions: inner speech, pronounced speech and visualized condition. All "
        "paradigms and the requested actions are explained in detail in the BCI Interaction Conditions Section. This "
        "dataset will allow future users to explore whether inner speech activates similar mechanisms as pronounced "
        "speech or whether it is closer to visualizing a spatial location or movement. Each participant performed "
        "between 475 and 570 trials in a single day recording, obtaining a dataset with more than 9hours of continuous "
        "EEG data recording, with over 5600 trials. At the beginning of each session, a fifteen seconds baseline was "
        "recorded where the participant was instructed to relax and stay as still as possible. Within each session, "
        "five stimulation runs were presented. Those runs correspond to the different proposed conditions: pronounced "
        "speech, inner speech and visualized condition (see Section BCI Interaction Conditions). At the beginning of "
        "each run, the condition was announced in the computer screen for a period of 3seconds. In all cases, the "
        "order of the runs was: one pronounced speech, two inner speech and two visualized conditions. A one minute "
        "break between runs was given (inter-run break).")
    citation: Optional[str] = """\
    @Article{Nieto2022,
    author={Nieto, Nicol{\'a}s
    and Peterson, Victoria
    and Rufiner, Hugo Leonardo
    and Kamienkowski, Juan Esteban
    and Spies, Ruben},
    title={Thinking out loud, an open-access EEG-based BCI dataset for inner speech recognition},
    journal={Scientific Data},
    year={2022},
    month={Feb},
    day={14},
    volume={9},
    number={1},
    pages={52},
    issn={2052-4463},
    doi={10.1038/s41597-022-01147-2},
    url={https://doi.org/10.1038/s41597-022-01147-2}
    }
    """

    filter_notch: float = 50.0

    dataset_name: Optional[str] = 'inner_speech'
    task_type: DatasetTaskType = DatasetTaskType.LINGUAL
    file_ext: str = 'bdf'

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

    valid_ratio: float = 0.2
    test_ratio: float = 0.2
    wnd_div_sec: int = 5
    suffix_path: str = 'Inner Speech'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: ['up', 'down', 'right', 'left'])


class InnerSpeechBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = InnerSpeechConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
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
        self.sub_meta = {
            'age': [56, 50, 34, 24, 30, 29, 26, 28, 35, 31],
            'sex': ['F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'M']
        }

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        filename = self._extract_file_name(file_path)
        subject, session, task, _ = filename.split('_')
        subject = int(subject.split('-')[1])
        session = int(session.split('-')[1])

        return {
            'subject': subject,
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        age = self.sub_meta['age'][info['subject'] - 1]
        sex = self.sub_meta['sex'][info['subject'] - 1]

        montage = 'biosemi128'

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

        # trigger correspondence table
        # https://www.nature.com/articles/s41597-022-01147-2/tables/4
        trig_inner_start = 22
        trig_visual_start = 23
        trig_up = 31
        trig_down = 32
        trig_right = 33
        trig_left = 34

        raw = self._read_raw_data(file_path, preload=True)
        sfreq = raw.info['sfreq']
        events = mne.find_events(raw, stim_channel='Status', verbose=False)

        inner_speech_start = np.where(events[:, 2] == trig_inner_start)[0][0]
        visual_speech_start = np.where(events[:, 2] == trig_visual_start)[0][0]

        annotations = []
        for name, trigger in zip(
                self.config.category,
                [trig_up, trig_down, trig_right, trig_left]
        ):
            index = np.where(events[:, 2] == trigger)[0]
            inner_mask: ndarray = (index > inner_speech_start) & (index < visual_speech_start)

            index = index[inner_mask]
            onsets = events[index, 0] / sfreq
            starts = onsets - 0.5
            ends = onsets + 4.5

            for i in range(len(onsets)):
                annotations.append((
                    name,
                    floor(starts[i].item() * 1000),
                    floor(ends[i].item() * 1000),
                ))

        return annotations


    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [self.mapping_dict[ch].upper() for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_bdf(file_path, preload=preload, verbose=verbose)
            return raw


if __name__ == "__main__":
    pass
    builder = InnerSpeechBuilder("finetune")
    builder.clean_disk_cache()
    builder.preproc(n_proc=6)
    builder.download_and_prepare(num_proc=6)
    dataset = builder.as_dataset()
    print(dataset)
