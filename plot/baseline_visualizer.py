import logging

import datasets
from torch import Tensor
from torch.utils.data import DataLoader

from baseline.abstract.classifier import MultiHeadClassifier
from common.config import AbstractConfig
from baseline.abstract.factory import ModelRegistry
from plot.base_visualizer import BaseVisualizer

logger = logging.getLogger()


class BaselineVisualizer(BaseVisualizer):
    def __init__(self, model_config: AbstractConfig, vis_args):
        if model_config.model.pretrained_path not in [None, ""]:
            raise ValueError('Visualizer only supports finetune models')

        super().__init__(model_config, vis_args)

    def build_model(self):
        logger.info(f"Building {self.vis_args.model_type} model for visualization")

        trainer_class = ModelRegistry.get_trainer_class(self.vis_args.model_type)
        self.trainer = trainer_class(self.cfg)

        self.trainer.setup_distributed()
        if self.cfg.multitask:
            self.trainer.collect_dataset_info(mixed=True)
        else:
            ds_name = next(iter(self.cfg.data.datasets.keys()))
            self.trainer.collect_dataset_info(mixed=False, ds_name=ds_name)
        self.model = self.trainer.setup_model()
        self.model.eval()

        if self.vis_args.ckpt_path:
            self.load_checkpoint()
        
        return self.model

    def create_dataloader(self, ds_name, ds_config) -> DataLoader:
        if self.vis_args.split == 'train':
            split = datasets.Split.TRAIN
        elif self.vis_args.split == 'valid':
            split = datasets.Split.VALIDATION
        elif self.vis_args.split == 'test':
            split = datasets.Split.TEST
        else:
            raise ValueError(f"Unsupported split: {self.vis_args.split}")
        dataloader, sampler = self.trainer.create_single_dataloader(ds_name, ds_config, split)
        return dataloader

    def extract_model_t_sne_features(self, ds_name: str) -> Tensor:
        model = self.get_model_from_ddp()

        classifier: MultiHeadClassifier = model.classifier
        return classifier.cls_feature.clone().detach().cpu()

    def find_target_layer(self):
        model = self.get_model_from_ddp()

        return model
