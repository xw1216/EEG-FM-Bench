import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
from mne.io import BaseRaw
from numpy import ndarray
from pandas import DataFrame
from scipy.io import loadmat

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class BCIC1AConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "This dataset from the Berlin BCI group contains EEG recordings for motor imagery tasks, "
        "where subjects imagined moving either their left hand, right hand, or foot. The data consists "
        "of two parts: calibration data (where subjects responded to visual cues) and evaluation data "
        "(where subjects responded to acoustic cues). The recording used 59 EEG channels, focused on "
        "sensorimotor areas.")

    citation: Optional[str] = """\
    @article{BLANKERTZ2007539,
    title = {The non-invasive Berlin Brain–Computer Interface: Fast acquisition of effective performance in untrained subjects},
    journal = {NeuroImage},
    volume = {37},
    number = {2},
    pages = {539-550},
    year = {2007},
    issn = {1053-8119},
    doi = {https://doi.org/10.1016/j.neuroimage.2007.01.051},
    url = {https://www.sciencedirect.com/science/article/pii/S1053811907000535},
    author = {Benjamin Blankertz and Guido Dornhege and Matthias Krauledat and Klaus-Robert Müller and Gabriel Curio},
    abstract = {Brain–Computer Interface (BCI) systems establish a direct communication channel from the brain to an output device. These systems use brain signals recorded from the scalp, the surface of the cortex, or from inside the brain to enable users to control a variety of applications. BCI systems that bypass conventional motor output pathways of nerves and muscles can provide novel control options for paralyzed patients. One classical approach to establish EEG-based control is to set up a system that is controlled by a specific EEG feature which is known to be susceptible to conditioning and to let the subjects learn the voluntary control of that feature. In contrast, the Berlin Brain–Computer Interface (BBCI) uses well established motor competencies of its users and a machine learning approach to extract subject-specific patterns from high-dimensional features optimized for detecting the user's intent. Thus the long subject training is replaced by a short calibration measurement (20 min) and machine learning (1 min). We report results from a study in which 10 subjects, who had no or little experience with BCI feedback, controlled computer applications by voluntary imagination of limb movements: these intentions led to modulations of spontaneous brain activity specifically, somatotopically matched sensorimotor 7–30 Hz rhythms were diminished over pericentral cortices. The peak information transfer rate was above 35 bits per minute (bpm) for 3 subjects, above 23 bpm for two, and above 12 bpm for 3 subjects, while one subject could achieve no BCI control. Compared to other BCI systems which need longer subject training to achieve comparable results, we propose that the key to quick efficiency in the BBCI system is its flexibility due to complex but physiologically meaningful features and its adaptivity which respects the enormous inter-subject variability.}
    }
    """

    filter_notch: float = 50.0
    orig_fs: float = 1000.0
    orig_lowpass: float = 200.0
    orig_highpass: float = 0.05

    dataset_name: Optional[str] = 'bcic_1a'
    task_type: DatasetTaskType = DatasetTaskType.MOTOR_IMAGINARY
    file_ext: str = 'mat'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        'sensorimotor': [
                         'AF3',       'AF4',
                 'F5','F3','F1','Fz','F2','F4','F6',
              'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
            'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
              'CP5','CP3','CP1','CPz','CP2','CP4','CP6',
                 'P5','P3','P1','Pz','P2','P4','P6',
                          'PO1',     'PO2',
                           'O1',     'O2',
        ]
    })

    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    wnd_div_sec: int = 8
    suffix_path: str = os.path.join('BCI Competition IV', '1a')
    scan_sub_dir: str = "BCICIV_1calib_1000Hz_mat"
    scan_eval_sub_dir: str = "BCICIV_1eval_1000Hz_mat"

    category: list[str] = field(default_factory=lambda: ['left', 'right', 'foot'])


class BCIC1ABuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = BCIC1AConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain', wnd_div_sec=8),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True, wnd_div_sec=6)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _walk_raw_data_files(self):
        scan_path = [os.path.join(self.config.raw_path, self.config.scan_sub_dir)]
        if not self.config.is_finetune:
            scan_path.append(os.path.join(self.config.raw_path, self.config.scan_eval_sub_dir))
        raw_data_files = []
        for path in scan_path:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(self.config.file_ext):
                        file_path = os.path.join(root, file)
                        raw_data_files.append(os.path.normpath(file_path))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        _, _, subject, _ = file_name.split('_')
        subject = subject[3:]
        return {
            'subject': subject,
            'session': 1,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        data = loadmat(file_path)
        fs = data['nfo']['fs'][0, 0].item()
        time = data['cnt'].shape[0] / fs

        info.update({
            'montage': 'sensorimotor',
            'time': time,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]

        data = loadmat(file_path)
        # time points are given in unit sample
        fs = data['nfo']['fs'][0, 0].item()
        tims = data['mrk']['pos'][0, 0].squeeze() / fs
        labels = data['mrk']['y'][0, 0].squeeze()
        cls = data['nfo']['classes'][0, 0].squeeze().tolist()

        annotations = []
        for i in range(len(labels)):
            c = cls[0][0] if labels[i] == -1 else cls[1][0]
            start = round(1000 * (tims[i].item()))
            end = round(start + 6 * 1000)
            assert c in self.config.category
            annotations.append((c, start, end))
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

    def _check_data_montage_channel(self, df: DataFrame, n_proc: Optional[int] = None):
        return df

    def _check_data_length(self, df: DataFrame):
        return df

    @staticmethod
    def _orig_ch_names():
        return [
                                    'AF3','AF4',
                        'F5','F3','F1','Fz','F2','F4','F6',
                    'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
            'CFC7','CFC5','CFC3','CFC1','CFC2','CFC4','CFC6','CFC8',
                    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
            'CCP7','CCP5','CCP3','CCP1','CCP2','CCP4','CCP6','CCP8',
                    'CP5','CP3','CP1','CPz','CP2','CP4','CP6',
                        'P5','P3','P1','Pz','P2','P4','P6',
                                    'PO1','PO2',
                                    'O1','O2',
        ]

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        data = loadmat(file_path)
        signal = data['cnt'].astype(np.float32).transpose(1, 0)
        fs = data['nfo']['fs'][0, 0].item()
        return self._convert_to_mne(signal, {'fs': fs})

    def _convert_to_mne(self, data: ndarray, info) -> mne.io.RawArray:
        ch_names = self._orig_ch_names()
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.config.orig_fs,
            ch_types=['eeg'] * len(ch_names))

        # mne expected unit is Volts, so turn 0.1 uV to V
        raw = mne.io.RawArray(data.astype(np.float32) / 1e7, info ,verbose=False)
        # raw.info['lowpass'] = self.config.orig_lowpass
        # raw.info['highpass'] = self.config.orig_highpass

        return raw

    def __fetch_mat_info(self, file_path):
        data = loadmat(file_path)
        signal: ndarray = 0.1 * data['cnt'].astype(np.float32).transpose(1, 0)

        cls = data['nfo']['classes'][0, 0].squeeze().tolist()
        chs = data['nfo']['clab'][0, 0].squeeze().tolist()
        cls = [item[0] for item in cls]
        chs = [item[0] for item in chs]

        fs = data['nfo']['fs'][0, 0].item()

        is_eval = self._extract_file_name(file_path).contains('eval')
        if not is_eval:
            tim = data['mrk']['pos'][0, 0].squeeze()
            label = data['mrk']['y'][0, 0].squeeze()
            return signal, fs, cls, chs, tim, label
        else:
            return signal, fs, cls, chs,


if __name__ == "__main__":
    builder = BCIC1ABuilder('finetune')
    builder.clean_disk_cache()
    builder.preproc()
    builder.download_and_prepare(num_proc=4)
    dataset = builder.as_dataset()
    print(dataset)
