import logging
import math
from typing import Optional

import numpy as np
import torch
import datasets
from torch import Tensor
from torch.utils.data import Dataset, Sampler, DataLoader

from common.config import AbstractConfig
from common.type import TrainStage
from data.processor.wrapper import load_concat_eeg_datasets

logger = logging.getLogger()


class DistributedGroupBatchSampler(Sampler):
    def __init__(
            self,
            dataset: Dataset,
            batch_size: int,
            sample_ratio: float = 1.0,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last=False,
    ):
        super().__init__()

        if torch.distributed.is_available():
            if num_replicas is None:
                num_replicas = torch.distributed.get_world_size()
            if rank is None:
                rank = torch.distributed.get_rank()
        else:
            if num_replicas is None or rank is None:
                raise ValueError("Must set num_replicas and rank when distributed package is not available")

        self.dataset = dataset
        self.sample_ratio = sample_ratio
        self.num_replicas = num_replicas
        self.rank = rank

        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        if len(dataset) < self.num_replicas:
            raise ValueError("Not enough data for training")
        self._adjust_batch_size(batch_size)

        self.epoch = 0
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        # group by record montage
        self.montage_groups: dict[str, Tensor] = {}
        self.n_total_batches = 0
        # pre gen batches
        self.all_batches = []

        self._group_by_montage()
        self._sampling_by_proportion()
        self._calculate_batch_numbers()
        self._pre_gen_all_batches()


    def _adjust_batch_size(self, batch_size):
        effective_batch_size = min(
            batch_size,
            math.floor(len(self.dataset) / self.num_replicas)
        )
        if effective_batch_size != batch_size:
            logger.warning(f'Adjust batch_size to {effective_batch_size} due to short of data')
            self.batch_size = effective_batch_size
        else:
            self.batch_size = batch_size

    def _group_by_montage(self):
        montage = np.array(self.dataset['montage'])
        unique_montages, inverse = np.unique(montage, return_inverse=True)
        self.montage_groups = {
            m: torch.where(torch.from_numpy(inverse) == idx)[0]
            for idx, m in enumerate(unique_montages)
        }

    def _sampling_by_proportion(self):
        if abs(self.sample_ratio - 1.0) < 1e-6:
            return

        n_original = sum(len(v) for v in self.montage_groups.values())
        if n_original == 0:
            return

        n_target = round(n_original * self.sample_ratio)
        montage_counts = {montage: len(indices) for montage, indices in self.montage_groups.items()}

        quotas = {}
        for montage, count in montage_counts.items():
            quota = (count / n_original) * n_target
            floor = math.floor(quota)
            remainder = quota - floor
            quotas[montage] = {'floor': floor, 'remainder': remainder}

        sum_floors = sum(q['floor'] for q in quotas.values())
        remaining = n_target - sum_floors

        # assign new samples for montage with a large remainder until reach to n_target
        sorted_montages = sorted(quotas.keys(), key=lambda x: quotas[x]['remainder'], reverse=True)
        for i in range(remaining):
            montage = sorted_montages[i]
            quotas[montage]['floor'] += 1

        new_montage_groups = {}
        for montage in self.montage_groups:
            sample_num = quotas[montage]['floor']
            indices = self.montage_groups[montage]
            if sample_num <= 0:
                continue
            if sample_num >= len(indices):
                new_montage_groups[montage] = indices
                continue
            else:
                perm = torch.randperm(len(indices), generator=self.generator)
                selected = indices[perm[:sample_num]]
                new_montage_groups[montage] = selected

        self.montage_groups = new_montage_groups

    def _calculate_batch_numbers(self):
        total_batches = 0
        self.group_batches_counter = {}
        for montage, indices in self.montage_groups.items():
            num_samples = len(indices)
            if self.drop_last:
                num_batches = num_samples // self.batch_size
            else:
                # this generates batches not full
                num_batches = math.ceil(num_samples / self.batch_size)

            self.group_batches_counter[montage] = num_batches
            total_batches += num_batches

        # make number of all batches can be evenly divisible by replicas
        remainder = total_batches % self.num_replicas
        if remainder == 0:
            self.n_total_batches = total_batches
        else:
            self.n_total_batches = total_batches - remainder

        assert self.n_total_batches >= self.num_replicas

        self.num_rank_batches = self.n_total_batches // self.num_replicas

    def _pre_gen_all_batches(self):
        all_batches: list[Tensor] = []
        for montage, indices in self.montage_groups.items():
            # shuffle in single montage
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.generator).tolist()
                indices = indices[perm]

            # gen batches in this montage
            batch_cnt = self.group_batches_counter[montage]
            indices_list = list(torch.split(indices, self.batch_size))[:batch_cnt]
            all_batches.extend(indices_list)

        if len(all_batches) != self.n_total_batches:
            logger.info(f'All batches num {len(all_batches)}, used batches num {self.n_total_batches}')
            logger.info(f'Last {len(all_batches) - self.n_total_batches} will be dropped.')
        else:
            logger.info(f'All batches num {len(all_batches)}')

        self.all_batches = all_batches[:self.n_total_batches]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed + self.epoch)

    def __iter__(self):
        # shuffle batches among various montage
        if self.shuffle:
            perm = torch.randperm(len(self.all_batches), generator=self.generator)
            all_batches = [self.all_batches[i] for i in perm]
        else:
            all_batches = self.all_batches

        return iter(all_batches[self.rank * self.num_rank_batches : (self.rank+1) * self.num_rank_batches])

    def __len__(self):
        return self.num_rank_batches


def assign_sampler_and_loader(dataset: datasets.Dataset, args: AbstractConfig, world_size: int, rank: int,):
    sampler = DistributedGroupBatchSampler(
        dataset=dataset,
        batch_size=args.data.batch_size,
        sample_ratio=args.data.sample_ratio,
        num_replicas=world_size,
        rank=rank,
        seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.data.num_loader_workers
    )
    return sampler, loader


def create_pretrain_concat_loader(
        args: AbstractConfig,
        world_size: int,
        rank: int,
        split: datasets.Split=datasets.Split.TRAIN
) -> tuple[Dataset, DistributedGroupBatchSampler, DataLoader]:
    if args.stage == TrainStage.PRETRAIN:
        dataset_list = args.data.datasets
        builder_configs = ['pretrain' for _ in range(len(args.data.datasets))]
    else:
        raise ValueError(f"Create pretrain loader for stage {args.stage}")

    dataset, _ = load_concat_eeg_datasets(
        dataset_list,
        builder_configs=builder_configs,
        split=split,
        weight_option='statistics'
    )
    sampler, loader = assign_sampler_and_loader(dataset, args, world_size, rank)
    return dataset, sampler, loader


def create_finetune_single_loader(
        args: AbstractConfig,
        ds_name: str,
        ds_config: str,
        world_size: int,
        rank: int,
        split: datasets.Split,
        add_ds_name: bool = False,
):
    if args.stage != TrainStage.FINETUNE:
        raise ValueError(f"Create finetune loader for stage {args.stage}")

    assert ds_name in args.finetune.datasets.keys() and ds_config == args.finetune.datasets[ds_name]
    dataset, weight = load_concat_eeg_datasets(
        [ds_name],
        builder_configs=[ds_config],
        split=split,
        weight_option='statistics',
        add_ds_name=add_ds_name
    )

    sampler, loader = assign_sampler_and_loader(dataset, args, world_size, rank)
    return dataset, sampler, loader, weight


def create_finetune_mixed_loader(
        args: AbstractConfig,
        world_size: int,
        rank: int,
        split: datasets.Split,
) -> tuple[Dataset, DistributedGroupBatchSampler, DataLoader, list[Tensor]]:
    if args.stage != TrainStage.FINETUNE:
        raise ValueError(f"Create finetune loader for stage {args.stage}")
    dataset_dict = args.finetune.datasets

    dataset, weights = load_concat_eeg_datasets(
        dataset_dict.keys(),
        builder_configs=dataset_dict.values(),
        split=split,
        weight_option=args.finetune.loss_weight_type,
        add_ds_name=True,
        cast_label=True
    )

    sampler, loader = assign_sampler_and_loader(dataset, args, world_size, rank)
    return dataset, sampler, loader, weights


def create_finetune_loader_list(
    args: AbstractConfig,
    world_size: int,
    rank: int,
    split: datasets.Split = datasets.Split.TRAIN
):
    if args.stage != TrainStage.FINETUNE:
        raise ValueError(f"Create finetune loader for stage {args.stage}")

    dataset_list, sampler_list, loader_list, weight_list = [], [], [], []
    for dataset_name, config_name in args.finetune.datasets.items():
        dataset, sampler, loader, distribution = create_finetune_single_loader(
            args, dataset_name, config_name, world_size, rank, split, add_ds_name=True)
        dataset_list.append(dataset)
        sampler_list.append(sampler)
        loader_list.append(loader)
        weight_list.append(distribution)

    return dataset_list, sampler_list, loader_list, weight_list


if __name__ == "__main__":
    data = load_concat_eeg_datasets(['seed_v'], ['finetune'], datasets.Split.TRAIN)
    sampler = DistributedGroupBatchSampler(data, 32, 0.8, 2, 0, drop_last=True,)

    loader = torch.utils.data.DataLoader(dataset=data, batch_sampler=sampler, num_workers=4)

    for batch in loader:
        print(batch)
