import logging
from typing import List

import torch
from torch import nn
import braindecode.models
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.classical import ClassicalTrainer
from baseline.eegnet.eegnet_config import EegNetConfig


logger = logging.getLogger('baseline')


class EegNetDataLoaderFactory(AbstractDataLoaderFactory):
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str]
    ) -> HFDataset:
        return dataset


class EegNetModel(nn.Module):
    def __init__(self, encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    def forward(self, batch):
        x = batch['data']
        logits = self.encoder(x)

        return logits

class EegNetTrainer(ClassicalTrainer):
    def __init__(self, cfg: EegNetConfig):
        super().__init__(cfg)

        self.dataloader_factory = EegNetDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed
        )

    def setup_model(self):
        logger.info(f"Setting up eegnet model architecture...")

        (ds_name, info) = next(iter(self.ds_info.items()))

        self.encoder = braindecode.models.EEGNetv1(
            n_outputs=info['n_class'],
            n_chans=info['n_ch'],
            n_times=info['wnd_sec'] * self.sfreq,
            sfreq=self.sfreq,
            add_log_softmax=False,
        )

        model = EegNetModel(self.encoder)
        model = model.to(self.device)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], find_unused_parameters=True
        )
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")

        self.model = model

        return model
