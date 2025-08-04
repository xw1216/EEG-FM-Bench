#!/usr/bin/env python3

import logging
import os

import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.eegpt.eegpt_adapter import EegptDataLoaderFactory
from baseline.eegpt.eegpt_config import EegptConfig
from baseline.eegpt.model import EEGTransformer
from baseline.utils.network import Conv1dWithConstraint


logger = logging.getLogger('baseline')


class EEGPTUnifiedModel(nn.Module):
    def __init__(self, encoder, classifier, grad_cam: bool, chan_conv=None):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.chan_conv = chan_conv

        self.grad_cam = grad_cam
        self.grad_cam_activation = None

    def forward(self, batch):
        x = batch['data']
        chans_id = batch['chans_id'][0]
        ds_name = batch['montage'][0].split('/')[0]

        # Apply channel convolution if available
        if self.chan_conv is not None:
            x = self.chan_conv(x)

        # Encoder forward pass
        features = self.encoder(x, chan_ids=chans_id)

        if self.grad_cam:
            self.grad_cam_activation = features.transpose(1, 2)

        features = features.reshape((features.shape[0],  features.shape[1], -1,))
        logits = self.classifier(features, ds_name)

        return logits


class EegptTrainer(AbstractTrainer):
    """
    EEGPT trainer that inherits from AbstractTrainer.
    """
    
    def __init__(self, cfg: EegptConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Initialize dataloader factory
        self.dataloader_factory = EegptDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed
        )
        
        # Model components
        self.target_encoder = None
        self.classifier = None
        self.chan_conv = None
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Model dimensions
        self.max_seq_length = 60 * 256  # 60 secs with 256Hz
        self.max_channels = 64

    def setup_model(self):
        """Setup EEGPT model architecture."""
        logger.info(f"Setting up EEGPT model architecture...")

        model_conf = self.cfg.model

        # Initialize EEGPT encoder
        self.target_encoder = EEGTransformer(
            img_size=[self.max_channels, self.max_seq_length],
            patch_size=model_conf.patch_size,
            patch_stride=model_conf.patch_stride,
            embed_num=model_conf.embed_num,
            embed_dim=model_conf.embed_dim,
            depth=model_conf.depth,
            num_heads=model_conf.num_heads,
            mlp_ratio=model_conf.mlp_ratio,
            drop_rate=model_conf.dropout_rate,
            attn_drop_rate=model_conf.attn_dropout_rate,
            drop_path_rate=model_conf.drop_path_rate,
            init_std=model_conf.init_std,
            qkv_bias=model_conf.qkv_bias,
            norm_layer=nn.LayerNorm,
        )

        # Channel adaptation layer (if configured)
        if self.cfg.model.use_channel_conv:
            self.chan_conv = Conv1dWithConstraint(
                self.cfg.model.conv_chan_dim,
                self.max_channels, 1, max_norm=1)

        # Create a classifier - always use multi-head for compatibility
        embed_dim = self.cfg.model.embed_dim * self.cfg.model.embed_num
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}

        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            mlp_dims=self.cfg.model.mlp_hidden_dim,
            head_configs=head_configs,
            dropout=self.cfg.model.dropout_rate,
            t_sne=self.cfg.model.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")

        self.load_checkpoint(self.cfg.model.pretrained_path)
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")

        model = EEGPTUnifiedModel(
            self.target_encoder,
            self.classifier,
            chan_conv=self.chan_conv,
            grad_cam=self.cfg.model.grad_cam,
        )
        model = model.to(self.device)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], find_unused_parameters=True
        )

        self.model = model

        return model

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return

        logger.info(f"Loading pretrained weights from: {checkpoint_path}")

        pretrain_ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Extract encoder weights
        target_encoder_state = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_state[k[15:]] = v

        # Load weights
        if target_encoder_state and self.target_encoder is not None:
            missing_keys, unexpected_keys = self.target_encoder.load_state_dict(target_encoder_state, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys in pretrained weights: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in pretrained weights: {unexpected_keys}")

            logger.info("Pretrained weights loaded successfully")
        else:
            logger.warning("No encoder weights found in checkpoint or encoder not initialized")


def main():
    """Main function to run EEGPT training."""
    import sys
    from omegaconf import OmegaConf
    
    if len(sys.argv) != 2:
        print("Usage: python eegpt_trainer.py config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load configuration
    file_cfg = OmegaConf.load(config_path)
    code_cfg = OmegaConf.create(EegptConfig().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg)
    config_dict = OmegaConf.to_container(merged_config, resolve=True)
    cfg = EegptConfig.model_validate(config_dict)
    
    # Create and run trainer
    trainer = EegptTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main() 