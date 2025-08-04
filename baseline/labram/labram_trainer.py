import logging
import os
from functools import partial

import torch
from torch import nn
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.labram.labram_adapter import LabramDataLoaderFactory
from baseline.labram.labram_config import LabramConfig
from baseline.labram.model import NeuralTransformer


logger = logging.getLogger('baseline')


class LabramUnifiedModel(nn.Module):
    """Unified Labram model wrapper for multitask training."""
    
    def __init__(self, encoder, classifier, grad_cam: bool = False):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

        self.grad_cam = grad_cam
        self.grad_cam_activation = None
        
    def forward(self, batch):
        # Labram expects data in shape (batch, n_channels, n_patches, patch_size)
        # Data comes in as (batch, n_channels, n_timepoints)
        x = batch['data']
        chans_id = batch['chans_id'][0]
        ds_name = batch['montage'][0].split('/')[0]
        batch_size, n_channels, n_timepoints = x.shape
        
        # Calculate patch parameters
        patch_size = self.encoder.patch_size
        n_patches = n_timepoints // patch_size
        
        # Reshape to patches
        data = x[:, :, :n_patches * patch_size]  # Trim to fit patches
        data = data.view(batch_size, n_channels, n_patches, patch_size)

        chans_id = nn.functional.pad(chans_id+1, (1, 0), value=0)

        # Get features from encoder
        features = self.encoder.forward_features(
            data,
            input_chans=chans_id,
            return_patch_tokens=True
        )

        if self.grad_cam:
            self.grad_cam_activation = features.transpose(1, 2)

        logits = self.classifier(features, ds_name)

        return logits


class LabramTrainer(AbstractTrainer):
    """
    LABRAM trainer that inherits from AbstractTrainer.
    """
    
    def __init__(self, cfg: LabramConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Initialize dataloader factory
        self.dataloader_factory = LabramDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed
        )
        
        # Model components
        self.encoder = None
        self.classifier = None
        
        # Loss function
        if self.cfg.training.label_smoothing > 0:
            self.loss_fn = LabelSmoothingCrossEntropy(smoothing=self.cfg.training.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def setup_model(self):
        """Setup Labram model architecture."""
        logger.info(f"Setting up Labram model architecture...")
        cfg = self.cfg.model

        self.encoder = NeuralTransformer(
                EEG_size=cfg.eeg_size,
                patch_size=cfg.patch_size,
                in_chans=cfg.in_chans,
                out_chans=cfg.out_chans,
                num_classes=0,
                embed_dim=cfg.embed_dim,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=partial(nn.LayerNorm, eps=1e-6),
                qk_scale=None,
                drop_rate=cfg.dropout_rate,
                attn_drop_rate=cfg.attn_dropout_rate,
                drop_path_rate=cfg.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=cfg.init_values,
                use_abs_pos_emb=cfg.use_abs_pos_emb,
                use_rel_pos_bias=cfg.use_rel_pos_bias,
                use_shared_rel_pos_bias=cfg.use_shared_rel_pos_bias,
                use_mean_pooling=cfg.use_mean_pooling,
                init_scale=cfg.init_scale,
        )

        # Create a classifier - always use multi-head for compatibility
        embed_dim = cfg.embed_dim
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}

        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            head_configs=head_configs,
            mlp_dims=cfg.mlp_hidden_dim,
            dropout=cfg.head_dropout,
            t_sne=cfg.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")

        self.load_checkpoint(self.cfg.model.pretrained_path)
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")

        model = LabramUnifiedModel(
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
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        encoder_state_dict = {}
        for k, v in checkpoint['model'].items():
            if k.startswith('student.'):
                encoder_state_dict[k[len('student.'):]] = v

        # Load weights into encoder
        if self.encoder is not None:
            missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = [], []
        
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        
        logger.info("Successfully loaded pretrained encoder weights")


def main():
    """Main function for standalone execution."""
    import sys
    from omegaconf import OmegaConf
    from common.path import get_conf_file_path
    from common.utils import setup_yaml
    
    setup_yaml()
    
    if len(sys.argv) < 2:
        raise ValueError("Please provide a config file path")
    
    # Load configuration
    conf_file_path = get_conf_file_path(sys.argv[1])
    file_cfg = OmegaConf.load(conf_file_path)
    
    # Create config object
    cfg = LabramConfig.model_validate(OmegaConf.to_container(file_cfg, resolve=True))
    
    # Create and run trainer
    trainer = LabramTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main() 