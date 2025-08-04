import logging
import os

import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.cbramod.cbramod_adapter import CBraModDataLoaderFactory
from baseline.cbramod.cbramod_config import CBraModConfig
from baseline.cbramod.model import CBraMod


logger = logging.getLogger('baseline')


class CBraModUnifiedModel(nn.Module):
    """Unified CBraMod model that combines encoder and classifier."""
    
    def __init__(self, encoder: CBraMod, classifier: MultiHeadClassifier, grad_cam: bool):
        super().__init__()
        self.patch_size = encoder.patch_size
        self.out_dim = encoder.out_dim
        self.encoder = encoder
        self.classifier = classifier

        self.grad_cam = grad_cam
        self.grad_cam_activation = None
    
    def forward(self, batch):
        """Forward pass through the unified model."""
        x = batch['data']# Shape: (batch_size, n_channels, n_timepoints)
        montage = batch['montage'][0]
        ds_name = montage.split('/')[0]
        
        # [batch_size, channels, patches, patch_size]
        batch_size, n_channels, n_timepoints = x.shape
        n_patches = n_timepoints // self.patch_size

        data = x.view(batch_size, n_channels, n_patches, self.patch_size)
        
        # [batch_size, n_channels, n_patches, out_dim]
        features = self.encoder(data)

        if self.grad_cam:
            self.grad_cam_activation = features

        features = features.view(batch_size, -1, self.out_dim)

        logits = self.classifier(features, ds_name)
        
        return logits


class CBraModTrainer(AbstractTrainer):
    """
    CBraMod trainer that inherits from AbstractTrainer.
    """
    
    def __init__(self, cfg: CBraModConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Initialize dataloader factory
        self.dataloader_factory = CBraModDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed
        )
        
        # Model components
        self.encoder = None
        self.classifier = None
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def setup_model(self):
        """Setup CBraMod model architecture."""
        logger.info(f"Setting up cbramod model architecture...")

        cfg = self.cfg.model

        # Initialize CBraMod encoder
        self.encoder = CBraMod(
            in_dim=cfg.in_dim,
            out_dim=cfg.out_dim,
            d_model=cfg.d_model,
            dim_ffn=cfg.dim_ffn,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
        )

        embed_dim = cfg.out_dim
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}

        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            mlp_dims=self.cfg.model.mlp_hidden_dim,
            head_configs=head_configs,
            dropout=self.cfg.model.head_dropout,
            t_sne=self.cfg.model.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")

        # Load checkpoint if specified
        if self.cfg.model.pretrained_path:
            self.load_checkpoint(self.cfg.model.pretrained_path)
        
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")

        model = CBraModUnifiedModel(
            self.encoder,
            self.classifier,
            grad_cam=self.cfg.model.grad_cam
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
        missing_keys, unexpected_keys = self.encoder.load_state_dict(pretrain_ckpt, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys in pretrained weights: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in pretrained weights: {unexpected_keys}")

        logger.info("Pretrained weights loaded successfully")


def main():
    """Main function to run CBraMod training."""
    import sys
    from omegaconf import OmegaConf
    
    if len(sys.argv) != 2:
        print("Usage: python CBraMod_trainer.py config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load configuration
    file_cfg = OmegaConf.load(config_path)
    code_cfg = OmegaConf.create(CBraModConfig().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg)
    config_dict = OmegaConf.to_container(merged_config, resolve=True)
    cfg = CBraModConfig.model_validate(config_dict)

    trainer = CBraModTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main() 