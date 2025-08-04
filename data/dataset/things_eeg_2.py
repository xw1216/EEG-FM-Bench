import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
from mne.io import BaseRaw, RawArray
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class ThingsEEG2Config(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "All images came from THINGS (Hebart et al., 2019), a database of 12 or more images of objects on a natural "
        "background for each of 1854 object concepts, where each concept (e.g., antelope, strawberry, t-shirt) belongs "
        "to one of 27 higher-level categories. The training image partition contains 1654 object concepts of 10 images "
        "each, for a total of 16,540 image conditions. The test image partition contains 200 object concepts of 1 image "
        "each, for a total of 200 image conditions. We presented participants with images of objects on a natural "
        "background using a RSVP paradigm. The paradigm consisted of rapid serial sequences of 20 images. Every sequence "
        "started with 750ms of blank screen, then each image was presented centrally for 100ms and a SOA of 200ms, and "
        "it ended with another 750ms of blank screen. After every rapid sequence there were up to 2s during which we "
        "instructed participants to first blink and then report, with a keypress, whether the target image appeared "
        "in the sequence.")
    citation: Optional[str] = """\
    @article{GIFFORD2022119754,
    title = {A large and rich EEG dataset for modeling human visual object recognition},
    journal = {NeuroImage},
    volume = {264},
    pages = {119754},
    year = {2022},
    issn = {1053-8119},
    doi = {https://doi.org/10.1016/j.neuroimage.2022.119754},
    url = {https://www.sciencedirect.com/science/article/pii/S1053811922008758},
    author = {Alessandro T. Gifford and Kshitij Dwivedi and Gemma Roig and Radoslaw M. Cichy},
    keywords = {Artificial neural networks, Computational neuroscience, Electroencephalography, 
    Open-access data resource, Neural encoding models, Visual object recognition},
    """

    filter_notch: float = 50.0

    dataset_name: Optional[str] = 'things_eeg_2'
    task_type: DatasetTaskType = DatasetTaskType.VISUAL
    file_ext: str = 'npy'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                                            'Fp1', 'Fp2',
                                 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
                          'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8',
            'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
                        'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
                        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                                 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                          'O1', 'Oz', 'O2',
        ]
    })

    valid_ratio: float = 0.20
    test_ratio: float = 0.20
    wnd_div_sec: int = 5
    suffix_path: str = 'THINGS-EEG-2'
    scan_sub_dir: str = "data"

    n_test_seq = 204
    n_train_seq = 840
    n_image_per_seq = 20
    t_pre_seq_start = 0.5
    t_post_seq_start = 4.5

    category: list[str] = field(default_factory=lambda: ['non-target', 'target'])


class ThingsEEG2Builder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = ThingsEEG2Config
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        filename = self._extract_file_name(file_path)
        splits = self._extract_middle_path(file_path, -3, -1)

        group = filename.split('_')[-1]
        subject = splits[0].split('-')[-1]
        session = splits[1].split('-')[-1]

        return {
            'subject': subject,
            'session': session,
            'group': group,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)

        data_dict = np.load(file_path, allow_pickle=True).item()
        time = data_dict['raw_eeg_data'].shape[1] / data_dict['sfreq']
        montage = '10_20'

        info.update({
            'montage': montage,
            'time': time,
        })

        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        behave_data_path = os.path.join(
            self.config.raw_path, 'behaviour', f'sub-{info["subject"]}',
            f'ses-{info["session"]}', 'behavioral_data.npy')

        behave_dict = np.load(behave_data_path, allow_pickle=True).item()
        response = behave_dict['response']
        target = behave_dict['target_present']

        if info['group'] == 'test':
            response = response[:self.config.n_test_seq]
            target = target[:self.config.n_test_seq]
        else:
            response = response[self.config.n_test_seq:]
            target = target[self.config.n_test_seq:]

        raw = self._read_raw_data(file_path)

        annotations = []
        for seq_idx, img_idx in enumerate(range(0, len(raw.annotations), self.config.n_image_per_seq)):
            if response[seq_idx] != target[seq_idx]:
                continue

            name = self.config.category[target[seq_idx]]
            onset = raw.annotations.onset[img_idx].item()
            start = round((onset - self.config.t_pre_seq_start) * 1000)
            end = round((onset + self.config.t_post_seq_start) * 1000)

            annotations.append((name, start, end))

        return annotations

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache.keys():
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.upper() for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        data_dict = np.load(file_path, allow_pickle=True).item()

        data = data_dict['raw_eeg_data']
        ch_types = data_dict['ch_types']
        sfreq = data_dict['sfreq']
        chs = data_dict['ch_names']
        chs = ['trigger' if item == 'stim' else item for item in chs]

        info = mne.create_info(
            ch_names=chs,
            ch_types=ch_types,
            sfreq=sfreq,
        )

        raw = self._convert_to_mne(data, info)
        return raw

    def _convert_to_mne(self, data: np.ndarray, info) -> mne.io.RawArray:
        # original data is stored in the Volts unit.
        raw: RawArray = mne.io.RawArray(data, info, verbose=False)

        stim_data, times = raw.get_data(picks=['trigger'], return_times=True)

        event_samples = np.where(stim_data[0] != 0)[0]
        event_ids = stim_data[0, event_samples].astype(int)

        onsets = times[event_samples]
        durations = [0.1] * len(onsets)
        descriptions = [str(eid) for eid in event_ids]

        annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions
        )

        raw.set_annotations(annotations)

        return raw

    def _resample_and_filter(self, data: BaseRaw):
        orig_fs = data.info['sfreq']
        if not self.config.is_notched:
            notch_freqs = np.arange(self.config.filter_notch, orig_fs / 2, self.config.filter_notch).tolist()
            data = data.notch_filter(freqs=notch_freqs, verbose=False)
        if orig_fs != self.config.fs:
            data = data.resample(sfreq=self.config.fs, verbose=False)
        return data


if __name__ == "__main__":
    builder = ThingsEEG2Builder("finetune")
    builder.clean_disk_cache()
    builder.preproc(n_proc=4)
    builder.download_and_prepare(num_proc=4)
    dataset = builder.as_dataset()
    print(dataset)
