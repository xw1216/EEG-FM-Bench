#!/usr/bin/env python3
"""
BIOT Trainer using Abstract Base Class

A unified BIOT trainer that inherits from AbstractTrainer and supports multiple training patterns.
BIOT uses STFT-based spectral features with linear attention transformer.
"""

import logging
import os
from typing import List

import torch
from torch import nn
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.classifier import MultiHeadClassifier, DynamicChannelConvRouter
from baseline.abstract.trainer import AbstractTrainer
from baseline.biot.biot_config import BiotConfig, BiotModelArgs
from baseline.biot.model import BIOTEncoder


logger = logging.getLogger('baseline')


class BiotDataLoaderFactory(AbstractDataLoaderFactory):
    """BIOT DataLoader factory that inherits from AbstractDataLoaderFactory."""

    def create_adapter(
            self,
            dataset: HFDataset,
            dataset_names: List[str],
            dataset_configs: List[str]
    ) -> HFDataset:
        return dataset


class BiotUnifiedModel(nn.Module):
    """Unified BIOT model with dynamic channel routing."""
    
    def __init__(self, encoder: BIOTEncoder, classifier, conv_router, grad_cam: bool):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.conv_router = conv_router

        self.grad_cam = grad_cam
        self.grad_cam_activation = None
        
    def forward(self, batch):
        x = batch['data']  # Shape: (batch_size, n_channels, n_timepoints)
        montage = batch['montage'][0]  # Get montage from batch
        ds_name = montage.split('/')[0]

        # trim data to times 200
        patch_size = self.encoder.n_fft
        n_patches = x.shape[2] // patch_size
        data = x[:, :, :n_patches * patch_size]

        # Route data through dynamic channel mapper
        mapped_data = self.conv_router(data, montage)

        if self.grad_cam:
            self.grad_cam_activation = mapped_data.transpose(1, 2)
        
        # Apply BIOT encoder (handles STFT and concatenation internally)
        features = self.encoder(mapped_data)
        
        # Apply classifier
        logits = self.classifier(features, ds_name)
        
        return logits


class BiotTrainer(AbstractTrainer):
    """BIOT trainer that inherits from AbstractTrainer."""
    
    def __init__(self, cfg: BiotConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Initialize dataloader factory
        self.dataloader_factory = BiotDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed
        )
        
        # Model components
        self.conv_router = None
        self.encoder = None
        self.classifier = None
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def setup_model(self):
        """Setup BIOT model architecture."""
        logger.info(f"Setting up BIOT model architecture...")

        model_cfg: BiotModelArgs = self.cfg.model

        logger.info(f"Target channels: {model_cfg.max_channels}")

        # Initialize BIOT encoder
        self.encoder = BIOTEncoder(
            emb_size=model_cfg.emb_size,
            heads=model_cfg.heads,
            depth=model_cfg.depth,
            n_channels=model_cfg.max_channels,
            n_fft=model_cfg.n_fft,
            hop_length=model_cfg.hop_length,
        )

        # Create classifier
        conv_configs = {ds_name: info['config'] for ds_name, info in self.ds_info.items()}
        self.conv_router = DynamicChannelConvRouter(
            conv_configs,
            target_channel=model_cfg.max_channels,
        )
        logger.info(f"Created dynamic convolution router: {list(conv_configs.keys())}")

        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        self.classifier = MultiHeadClassifier(
            embed_dim=model_cfg.emb_size,
            mlp_dims=model_cfg.mlp_hidden_dim,
            head_configs=head_configs,
            dropout=model_cfg.head_dropout,
            average_pooling=False,
            t_sne=model_cfg.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")

        # Load pretrained weights if available
        if self.cfg.model.pretrained_path:
            self.load_checkpoint(self.cfg.model.pretrained_path)

        model = BiotUnifiedModel(
            encoder=self.encoder,
            classifier=self.classifier,
            conv_router=self.conv_router,
            grad_cam=self.cfg.model.grad_cam,
        )
        model = model.to(self.device)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], find_unused_parameters=True
        )
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")

        self.model = model

        return model

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return

        logger.info(f"Loading pretrained weights from: {checkpoint_path}")

        pretrain_ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        missing_keys, unexpected_keys = self.encoder.load_state_dict(pretrain_ckpt, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")

        logger.info("Successfully loaded pretrained encoder weights")
