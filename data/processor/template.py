import logging
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


@dataclass
class TemplateConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("0.0.0")
    description: Optional[str] = ""
    citation: Optional[str] = """\
    @Article{,
    AUTHOR = {},
    TITLE = {},
    JOURNAL = {},
    VOLUME = {},
    YEAR = {},
    NUMBER = {},
    ARTICLE-NUMBER = {},
    URL = {},
    ISSN = {},
    ABSTRACT = {},
    DOI = {}}
    """

    filter_notch: float = 50.0
    is_notched: bool = True

    dataset_name: Optional[str] = 'template'
    task_type: DatasetTaskType = DatasetTaskType.UNKNOWN
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
            'Fp1',
            'Fp2',
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 10
    suffix_path: str = 'template'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: ['class 1', 'class 2'])


class TemplateBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = TemplateConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        pass

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        pass

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        pass

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        return [('default', 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        return self.config.montage[montage]


if __name__ == "__main__":
    pass
    # builder = TemplateBuilder("pretrain")
    # builder.preproc()
    # builder.download_and_prepare(num_proc=1)
    # dataset = builder.as_dataset()
    # print(dataset)
