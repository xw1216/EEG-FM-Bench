"""
EEGPT Configuration that inherits from AbstractConfig.
"""

from typing import Dict, Optional, List
from pydantic import Field

from common.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class EegptDataArgs(BaseDataArgs):
    """EEGPT data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class EegptModelArgs(BaseModelArgs):
    """EEGPT model configuration."""
    # Pretrained model path
    pretrained_path: str = "/path/to/your/code/baseline/eegpt/ckpt/eegpt_mcae_58chs_4s_large4E.ckpt"

    # Architecture parameters
    patch_size: int = 64
    patch_stride: Optional[int] = 32
    embed_num: int = 4
    embed_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # Regularization
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    drop_path_rate: float = 0.1

    # Model initialization
    init_std: float = 0.02
    qkv_bias: bool = True

    # Channel adaptation
    use_channel_conv: bool = False
    conv_chan_dim: int = 22

    # Classification head
    linear_probe1_dim: int = 16
    linear_probe1_max_norm: float = 1.0
    linear_probe2_max_norm: float = 0.25
    head_dropout: float = 0.3
    mlp_hidden_dim: list[int] = Field(default_factory=lambda: [128])


class EegptTrainingArgs(BaseTrainingArgs):
    """EEGPT training configuration."""
    max_epochs: int = 100

    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_schedule: str = "onecycle"  # 'onecycle' or 'cosine'
    max_lr: float = 1e-4
    encoder_lr_scale: float = 0.1
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2  # For OneCycleLR
    min_lr: float = 1e-6  # For CosineAnnealingLR

    use_amp: bool = True
    # Training options
    freeze_encoder: bool = True

    label_smoothing: float = 0.0


class EegptLoggingArgs(BaseLoggingArgs):
    """EEGPT logging configuration."""
    experiment_name: str = "eegpt"
    output_dir: str = "/path/to/your/code/baseline/eegpt/log"
    ckpt_dir: str = "/path/to/your/code/baseline/eegpt/ckpt"

    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"  # 'wandb', 'comet', or 'both'
    project: Optional[str] = "eegpt"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: ["eegpt"])

    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class EegptConfig(AbstractConfig):
    """EEGPT configuration that extends AbstractConfig."""
    
    model_type: str = "eegpt"
    
    data: EegptDataArgs = Field(default_factory=EegptDataArgs)
    model: EegptModelArgs = Field(default_factory=EegptModelArgs)
    training: EegptTrainingArgs = Field(default_factory=EegptTrainingArgs)
    logging: EegptLoggingArgs = Field(default_factory=EegptLoggingArgs)

    def validate_config(self) -> bool:
        """Validate EEGPT-specific configuration."""
        # Check patch size requirements
        if self.model.patch_size <= 0:
            return False
        
        # Check embed dimensions
        if self.model.embed_dim % self.model.num_heads != 0:
            return False
        
        # Check learning rate schedule
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False
            
        return True 