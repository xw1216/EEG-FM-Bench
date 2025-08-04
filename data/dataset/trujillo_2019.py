from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class Trujillo2019Config(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "This is the raw EEG data for the study. Data is in BioSemi Data Format (BDF). Files with only 'II' in "
        "the file name were recorded during the reported Information-Integration categorization task; 'RB-II' "
        "files were recorded during the reported multidimensional Rule-Based categorization task. Dataset"
        "available at https://doi.org/10.18738/T8/SS2NHB")
    citation: Optional[str] = """\
    @ARTICLE{10.3389/fnins.2019.01292,
    AUTHOR={Trujillo, Logan T. },
    TITLE={Mental Effort and Information-Processing Costs Are Inversely Related to Global Brain Free Energy During Visual Categorization},
    JOURNAL={Frontiers in Neuroscience},
    VOLUME={13},
    YEAR={2019},
    URL={https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.01292},
    DOI={10.3389/fnins.2019.01292},
    ISSN={1662-453X},}
    """

    filter_notch: float = 60.0

    dataset_name: Optional[str] = 'trujillo_2019'
    task_type: DatasetTaskType = DatasetTaskType.VISUAL
    file_ext: str = 'bdf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_10': [
                                     'Fp1', 'Fpz', 'Fp2',
                              'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
             'FT7', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8',
                       'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
                    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
              'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
                              'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                      'O1', 'Oz', 'O2',
                                            'Iz',
        ]
    })

    valid_ratio: float = 0.0625
    test_ratio: float = 0.0
    wnd_div_sec: int = 30
    suffix_path: str = "Raw EEG Data Trujillo 2019"
    scan_sub_dir: str = "dataverse_files"

    category: list[str] = field(default_factory=lambda: [])


class Trujillo2019Builder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = Trujillo2019Config
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        file_name = self._extract_file_name(file_path)
        session, subject = file_name.split('_')[-2:]
        subject = subject[1:]
        if session == 'II':
            session = 1
        elif session == 'RBII':
            session = 2
        else:
            raise ValueError(f"Invalid session name: {session}")

        return {
            'subject': int(subject),
            'session': session,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        with self._read_raw_data(file_path, preload=False, verbose=False) as raw:
            time = raw.duration
            date = raw.info['meas_date']

        info.update({
            'montage': '10_10',
            'time': time,
            'date': date.strftime("%d%m%Y"),
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


if __name__ == "__main__":
    builder = Trujillo2019Builder()
    builder.clean_disk_cache()
    builder.preproc()
    builder.download_and_prepare(num_proc=2)
    dataset = builder.as_dataset()
    print(dataset)
