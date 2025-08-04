"""
BENDR Configuration that inherits from AbstractConfig.
"""

from typing import Dict, Optional, List
from pydantic import Field

from common.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class BendrDataArgs(BaseDataArgs):
    """BENDR data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 64
    num_workers: int = 1


class BendrModelArgs(BaseModelArgs):
    """BENDR model configuration."""
    # Pretrained model paths - dict with separate encoder/contextualizer paths
    pretrained_path: Optional[str] = None
    pretrained_conv_path: Optional[str] = None
    
    # BENDR encoder parameters
    emb_dim: int = 512
    conv_width: List[int] = Field(default_factory=lambda: [3, 2, 2, 2, 2, 2])
    conv_drop_rate: float = 0.0
    conv_proj_head: bool = False
    conv_stride: List[int] = Field(default_factory=lambda: [3, 2, 2, 2, 2, 2])
    
    # BENDR contextualizer parameters
    ffn_dim: int = 3076
    heads: int = 8
    context_layers: int = 8
    context_drop_rate: float = 0.15
    activation: str = 'gelu'
    position_encoder: int = 25
    layer_drop: float = 0.0
    
    # Masking parameters
    mask_p_t: float = 0.1
    mask_p_c: float = 0.004
    mask_t_span: int = 6
    
    # Classification head
    head_dropout: float = 0.1
    mlp_hidden_dim: list[int] = Field(default_factory=lambda: [128])
    
    # Model dimensions
    max_channels: int = 20


class BendrTrainingArgs(BaseTrainingArgs):
    """BENDR training configuration."""
    max_epochs: int = 50
    
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_schedule: str = "cosine"  # 'onecycle' or 'cosine'
    max_lr: float = 1e-4
    encoder_lr_scale: float = 0.1
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2  # For OneCycleLR
    min_lr: float = 1e-6  # For CosineAnnealingLR
    
    use_amp: bool = True
    freeze_encoder: bool = True
    
    # BENDR-specific training options
    finetuning: bool = True  # Enable finetuning mode for contextualizer


class BendrLoggingArgs(BaseLoggingArgs):
    """BENDR logging configuration."""
    experiment_name: str = "bendr"
    output_dir: str = "/path/to/your/code/bendr/log"
    ckpt_dir: str = "/path/to/your/code/bendr/ckpt"
    
    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"  # 'wandb', 'comet', or 'both'
    project: Optional[str] = "bendr"
    entity: Optional[str] = None
    
    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])
    
    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class BendrConfig(AbstractConfig):
    """BENDR configuration that extends AbstractConfig."""
    
    model_type: str = "bendr"
    
    data: BendrDataArgs = Field(default_factory=BendrDataArgs)
    model: BendrModelArgs = Field(default_factory=BendrModelArgs)
    training: BendrTrainingArgs = Field(default_factory=BendrTrainingArgs)
    logging: BendrLoggingArgs = Field(default_factory=BendrLoggingArgs)
    
    def validate_config(self) -> bool:
        """Validate BENDR-specific configuration."""
        # Check encoder dimensions
        if self.model.emb_dim <= 0:
            return False
        
        # Check encoder width and stride consistency
        if len(self.model.conv_width) != len(self.model.conv_stride):
            return False
        
        # Check contextualizer parameters
        if self.model.emb_dim % self.model.heads != 0:
            return False
        
        # Check learning rate schedule
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False
        
        # Check masking parameters
        if self.model.mask_p_t < 0 or self.model.mask_p_c < 0:
            return False
        
        return True 