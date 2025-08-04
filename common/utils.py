import random
from enum import Enum
from typing import Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torch import nn, Tensor

from common.type import PriorType


def split_array_equal_part(arr: Union[list[int], Tensor] , k: int):
    return [arr[i:i + k] for i in range(0, len(arr), k)]


def update_conf_dict(conf: DictConfig, src: str, dst: str):
    conf = conf.copy()
    conf[dst] = conf[src]
    return conf


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_num_params(model: nn.Module) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    param = {n: p.numel() for n, p in model.named_parameters()}
    return sum(param.values())


def yaml_enum_represent(dumper: yaml.Dumper, enum_obj: Enum) -> yaml.ScalarNode:
    return dumper.represent_scalar(
        tag='tag:yaml.org,2002:str',
        value=enum_obj.value
    )

def setup_yaml():
    yaml.representer.Representer.add_multi_representer(Enum, yaml_enum_represent)


class ElectrodeSet:
    """
    A class to represent a set of EEG electrodes based on the 10-10 system.
    https://www.fieldtriptoolbox.org/assets/img/template/layout/eeg1010.lay.png
    """
    Layout = '10-10'
    Count = 90
    Electrodes = [
                                   'FP1', 'FPZ', 'FP2',
       'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
      'T1', 'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', 'T2',
       'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
      'A1', 'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', 'A2',
       'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
            'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
       'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
                                    'O1', 'OZ', 'O2',
                                    'I1', 'IZ', 'I2',
    ]

    def __init__(self):
        self.electrode_dict = {electrode: i for i, electrode in enumerate(self.Electrodes)}
        self.index_dict = {i: electrode for i, electrode in enumerate(self.Electrodes)}

    def __len__(self):
        return self.Count

    def get_electrodes_index(self, electrodes: list[str]) -> np.ndarray:
        return np.array([self.electrode_dict[electrode] for electrode in electrodes], dtype=np.int32)

    def get_electrodes_name(self, electrodes: list[int]) -> list[str]:
        return [self.index_dict[electrode] for electrode in electrodes]

    def _create_boolean_matrix(self, input_dict):
        element_to_index = {element: idx for idx, element in enumerate(self.Electrodes)}

        bool_matrix = []

        for key, elements in input_dict.items():
            bool_list = [False] * len(self.Electrodes)

            for element in elements:
                if element in element_to_index:
                    idx = element_to_index[element]
                    bool_list[idx] = True

            bool_matrix.append(bool_list)

        bool_matrix = np.array(bool_matrix, dtype=np.bool)
        return bool_matrix


def clean_torch_distributed(local_rank: int):
    # deepspeed env can also be cleand
    # torch.distributed.barrier(device_ids=[local_rank])
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
