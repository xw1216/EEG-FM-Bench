import logging
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class OpenMiirConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("0.0.0")
    description: Optional[str] = (
        "Music imagery information retrieval (MIIR) systems may one day be able to recognize a song just as we think "
        "of it. As a step towards such technology, we are presenting a public domain dataset of electroencephalography "
        "(EEG) recordings taken during music perception and imagination. We acquired this data during an ongoing study "
        "that so far comprised 10 subjects listening to and imagining 12 short music fragments - each 7s-16s long - "
        "taken from well-known pieces. These stimuli were selected from different genres and systematically span "
        "several musical dimensions such as meter, tempo and the presence of lyrics. This way, various retrieval "
        "and classification scenarios can be addressed. The dataset is primarily aimed to enable music information "
        "retrieval researchers interested in these new MIIR challenges to easily test and adapt their existing "
        "approaches for music analysis like fingerprinting, beat tracking or tempo estimation on this new kind of data. "
        "We also hope that the OpenMIIR dataset will facilitate a stronger interdisciplinary collaboration between "
        "music information retrieval researchers and neuroscientists.")
    citation: Optional[str] = """\
    @inproceedings{stober2015towards,
    title={Towards music imagery information retrieval: Introducing the OpenMIIR dataset of EEG recordings 
    from music perception and imagination.},
    author={Stober, Sebastian and Sternin, Avital and Owen, Adrian M and Grahn, Jessica A},
    booktitle={ISMIR},
    pages={763--769},
    year={2015}
    }
    """

    filter_notch: float = 60.0

    dataset_name: Optional[str] = 'open_miir'
    task_type: DatasetTaskType = DatasetTaskType.AUDIO
    file_ext: str = 'fif'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                                   'Fp1', 'Fpz', 'Fp2',
                            'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
                  'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
              'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
                  'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
              'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
                            'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                    'O1', 'Oz', 'O2',
                                          'Iz',
            ]
    })

    valid_ratio: float = 0.23
    test_ratio: float = 0.0
    wnd_div_sec: int = 10
    suffix_path: str = 'OpenMIIR'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: [])


class OpenMiirBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = OpenMiirConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        filename = self._extract_file_name(file_path)
        subject = filename.split('-')[0][1:]

        return {
            'subject': subject,
            'session': 1,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        montage = '10_20'

        with self._read_raw_data(file_path) as raw:
            time = raw.duration

        info.update({
            'time': time,
            'montage': montage,
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
    builder = OpenMiirBuilder("pretrain")
    builder.clean_disk_cache()
    builder.preproc(n_proc=1)
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
