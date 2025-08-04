import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne_bids
from mne_bids import BIDSPath
from pandas import DataFrame
from tqdm import tqdm

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class ChiscoConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.1.0")
    description: Optional[str] = (
        "We present the Chinese Imagined Speech Corpus (Chisco), including over 20,000 sentences of high-density EEG "
        "recordings of imagined speech from healthy adults. Each subject’s EEG data exceeds 900 minutes, representing "
        "the largest dataset per individual currently available for decoding neural language to date. Furthermore, "
        "the experimental stimuli include over 6,000 everyday phrases across 39 semantic categories, covering nearly "
        "all aspects of daily language. we manually selected daily expressions from Chinese social media platform Weibo, "
        "public datasets ROCstory20 and Dailydialog21. These expressions were initially categorized into 39 types using "
        "a combination of machine learning clustering algorithms and manual annotation by human experts. Each trial "
        "started with a reading phase, during which the subjects were presented with a single sentence on the computer "
        "screen for 5,000ms and were instructed to saliently read it. The details of the presented sentences are "
        "specified in the “Text Materials” section. Following the reading phase, a blank screen was displayed for "
        "3,300ms, during which the subjects were required to imagine the sentence in mind at a constant speed. During "
        "this stage, the subjects need to imagine themselves speaking and not making any movements in their "
        "throat or limbs.")
    citation: Optional[str] = """\
    @Article{Zhang2024,
    author={Zhang, Zihan and Ding, Xiao and Bao, Yu and Zhao, Yi and Liang, Xia and Qin, Bing and Liu, Ting},
    title={Chisco: An EEG-based BCI dataset for decoding of imagined speech},
    journal={Scientific Data},
    year={2024},
    month={Nov},
    day={21},
    volume={11},
    number={1},
    pages={1265},
    issn={2052-4463},
    doi={10.1038/s41597-024-04114-1},
    url={https://doi.org/10.1038/s41597-024-04114-1}
    }
    """

    filter_notch: float = 50.0

    dataset_name: Optional[str] = 'chisco'
    task_type: DatasetTaskType = DatasetTaskType.LINGUAL
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        'quik_cap_128': [
                                        'FP1', 'FPz', 'Fp2',
                                 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
                       'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT9', 'Ft7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
                 'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
                           'TP7', 'CP3', 'CP1',      'CP2', 'CP4', 'TP8',
                        'P9', 'P7', 'P5', 'P3',      'P4', 'P8', 'P6', 'P10',
                          'PO9', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO10',
                                                'Oz',
                                                'Iz',
        ]
    })

    valid_ratio: float = 0.2
    test_ratio: float = 0.0
    wnd_div_sec: int = 30
    suffix_path: str = 'Chisco'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: [])


class ChiscoBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = ChiscoConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _walk_raw_data_files(self):
        # !IMPORTANT derivatives folder should be moved out from the scan folder to avoid duplicate files
        logger.info('Parsing BIDS path...')
        scan_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        bids_path = BIDSPath(datatype="eeg", root=scan_path, extension=f".{self.config.file_ext}")
        all_paths = bids_path.match()

        raw_data_files = [str(path.fpath) for path in all_paths]
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        bids_path = mne_bids.get_bids_path_from_fname(file_path)
        return {
            'subject': bids_path.subject,
            'session': bids_path.session,
            'run': bids_path.run,
            'task': bids_path.task,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)

        montage = 'quik_cap_128'
        with self._read_raw_data(file_path) as raw:
            time = raw.duration

        info.update({
            'montage': montage,
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

        chs: list[str] = self.config.montage[montage]
        chs_std = [ch.upper() for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std


if __name__ == "__main__":
    builder = ChiscoBuilder("pretrain")
    builder.clean_disk_cache()
    builder.preproc(n_proc=1)
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
