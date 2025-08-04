import logging
from typing import Type

import datasets
import torch
from datasets import Dataset, concatenate_datasets, Value
from torch import Tensor
from torch.utils.data import DataLoader

from data.dataset.adftd import AdftdBuilder
from data.dataset.bcic.bcic_1a import BCIC1ABuilder
from data.dataset.bcic.bcic_2020_3 import BCIC2020ImagineBuilder
from data.dataset.bcic.bcic_2a import BCIC2ABuilder
from data.dataset.brain_lat import BrainLatBuilder
from data.dataset.chisco import ChiscoBuilder
from data.dataset.emobrain import EmobrainBuilder
from data.dataset.grasp_and_lift import GraspAndLiftBuilder
from data.dataset.hbn import HBNBuilder
from data.dataset.hmc import HMCBuilder
from data.dataset.inner_speech import InnerSpeechBuilder
from data.dataset.inria_bci import InriaBciBuilder
from data.dataset.mimul_11 import Mimul11Builder
from data.dataset.motor_mv_img import MotorMoveImagineBuilder
from data.dataset.openmiir import OpenMiirBuilder
from data.dataset.seeds.seed import SeedBuilder
from data.dataset.seeds.seed_fra import SeedFraBuilder
from data.dataset.seeds.seed_ger import SeedGerBuilder
from data.dataset.seeds.seed_iv import SeedIVBuilder
from data.dataset.seeds.seed_v import SeedVBuilder
from data.dataset.seeds.seed_vii import SeedVIIBuilder
from data.dataset.siena_scalp import SienaScalpBuilder
from data.dataset.spis_resting_state import SpisRestingStateBuilder
from data.dataset.target_versus_non import TargetVersusNonBuilder
from data.dataset.things_eeg import ThingsEEGBuilder
from data.dataset.things_eeg_2 import ThingsEEG2Builder
from data.dataset.trujillo_2017 import Trujillo2017Builder
from data.dataset.trujillo_2019 import Trujillo2019Builder
from data.dataset.tue.tuab import TuabBuilder
from data.dataset.tue.tuar import TuarBuilder
from data.dataset.tue.tueg import TuegBuilder
from data.dataset.tue.tuep import TuepBuilder
from data.dataset.tue.tuev import TuevBuilder
from data.dataset.tue.tusl import TuslBuilder
from data.dataset.tue.tusz import TuszBuilder
from data.dataset.workload import WorkloadBuilder
from data.processor.builder import EEGDatasetBuilder, EEGConfig


log = logging.getLogger()


DATASET_SELECTOR: dict[str, Type[EEGDatasetBuilder]] = {
    'tuab': TuabBuilder,
    'tuar': TuarBuilder,
    'tueg': TuegBuilder,
    'tuep': TuepBuilder,
    'tuev': TuevBuilder,
    'tusl': TuslBuilder,
    'tusz': TuszBuilder,
    'seed': SeedBuilder,
    'seed_fra': SeedFraBuilder,
    'seed_ger': SeedGerBuilder,
    'seed_iv': SeedIVBuilder,
    'seed_v': SeedVBuilder,
    'seed_vii': SeedVIIBuilder,
    'bcic_1a': BCIC1ABuilder,
    'bcic_2a': BCIC2ABuilder,
    'bcic_2020_3': BCIC2020ImagineBuilder,
    'emobrain': EmobrainBuilder,
    'grasp_and_lift': GraspAndLiftBuilder,
    'hmc': HMCBuilder,
    'inria_bci': InriaBciBuilder,
    'motor_mv_img': MotorMoveImagineBuilder,
    'siena_scalp': SienaScalpBuilder,
    'spis_resting_state': SpisRestingStateBuilder,
    'target_versus_non': TargetVersusNonBuilder,
    'trujillo_2017': Trujillo2017Builder,
    'trujillo_2019': Trujillo2019Builder,
    'workload': WorkloadBuilder,
    'hbn': HBNBuilder,
    'adftd': AdftdBuilder,
    'brain_lat': BrainLatBuilder,
    'things_eeg': ThingsEEGBuilder,
    'things_eeg_2': ThingsEEG2Builder,
    'mimul_11': Mimul11Builder,
    'inner_speech': InnerSpeechBuilder,
    'chisco': ChiscoBuilder,
    'open_miir': OpenMiirBuilder,
}

def get_dataset_patch_len(dataset_name: str, config_name: str) -> int:
    config: EEGConfig = DATASET_SELECTOR[dataset_name].builder_configs.get(config_name)
    return config.wnd_div_sec

def get_dataset_n_class(dataset_name: str, config_name: str) -> int:
    config: EEGConfig = DATASET_SELECTOR[dataset_name].builder_configs.get(config_name)
    return len(config.category)

def get_dataset_category(dataset_name: str, config_name: str) -> list[str]:
    config: EEGConfig = DATASET_SELECTOR[dataset_name].builder_configs.get(config_name)
    return config.category

def get_dataset_montage(dataset_name: str, config_name: str) -> dict[str, list[str]]:
    builder_cls = DATASET_SELECTOR[dataset_name]
    builder: EEGDatasetBuilder = builder_cls(config_name=config_name)
    montage_names = builder.config.montage.keys()

    montages: dict[str, list[str]] = dict()
    for montage_name in montage_names:
        montages[f'{dataset_name}/{montage_name}'] = builder.standardize_chs_names(montage_name)

    return montages


def load_concat_eeg_datasets(
        dataset_names: list[str],
        builder_configs: list[str],
        split: datasets.NamedSplit = datasets.Split.TRAIN,
        weight_option: str = 'statistics',
        add_ds_name: bool = False,
        cast_label: bool = False,
) -> tuple[Dataset, list[Tensor]]:
    dataset_list = []
    weight_list = []
    for ds_name, ds_config in zip(dataset_names, builder_configs):
        try:
            builder_cls = DATASET_SELECTOR[ds_name]
            builder = builder_cls(config_name=ds_config)
            # noinspection PyTypeChecker
            dataset: Dataset = builder.as_dataset(split=split)
            if add_ds_name:
                dataset = dataset.add_column('ds_name', [ds_name for _ in range(len(dataset))])

            if 'label' in dataset.column_names:
                label = torch.tensor(dataset['label'], dtype=torch.int32)
                label_cnt = torch.bincount(label, minlength=get_dataset_n_class(ds_name, ds_config))
                log.info(f'Sample distribution for {ds_name}-{ds_config} {split}: {label_cnt}')
                weight = calc_distribution_weight(len(dataset), label_cnt, weight_option)

                weight_list.append(weight)

                if cast_label:
                    dataset = dataset.cast_column('label', Value('int64'))

            dataset_list.append(dataset)
        except KeyError:
            log.error(f'Dataset {ds_name} not found')

    combined_dataset: Dataset = concatenate_datasets(dataset_list)
    # combined_dataset = combined_dataset.flatten_indices()
    return combined_dataset.with_format('torch'), weight_list

def calc_distribution_weight(n: int, label_cnt: Tensor, option: str):
    if option == 'statistics':
        return label_cnt
    elif option == 'sqrt':
        return n / torch.sqrt(label_cnt.float() + 1)
    elif option == 'log':
        return n / torch.log(label_cnt.float() + 1)
    elif option == 'absolute':
        return n / label_cnt.float()
    else:
        raise ValueError(f'Unknown option {option}')



if __name__ == '__main__':
    # data = load_concat_eeg_datasets(['seed_v', 'tuab'])
    data, distribution = load_concat_eeg_datasets(['tuab'], ['finetune'])
    loader = DataLoader(data, batch_size=32)

    for batch in loader:
        pass


