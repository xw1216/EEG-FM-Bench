"""
LABRAM Configuration that inherits from AbstractConfig.
"""

from typing import Dict, Optional, List
from pydantic import Field

from common.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class LabramDataArgs(BaseDataArgs):
    """LABRAM data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 64
    num_workers: int = 4


class LabramModelArgs(BaseModelArgs):
    """LABRAM model configuration."""
    # Pretrained model path
    pretrained_path: str = "/path/to/your/code/baseline/labram/ckpt/labram-base.pth"

    # Model architecture
    model_name: str = "labram_base_patch200_200"
    
    # Input configuration
    eeg_size: int = 2000
    patch_size: int = 200

    in_chans: int = 1
    out_chans: int = 8
    
    # Architecture parameters
    embed_dim: int = 200
    depth: int = 12
    num_heads: int = 10
    mlp_ratio: float = 4.0

    # Regularization
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0
    drop_path_rate: float = 0.1

    # Model initialization
    init_values: float = 0.1
    init_scale: float = 0.001
    layer_scale_init_value: float = 0.1
    qkv_bias: bool = False

    # Position embeddings
    use_abs_pos_emb: bool = True
    use_rel_pos_bias: bool = False
    use_shared_rel_pos_bias: bool = False
    use_mean_pooling: bool = True

    # Classification head
    head_dropout: float = 0.1
    mlp_hidden_dim: list[int] = Field(default_factory=lambda: [128])


class LabramTrainingArgs(BaseTrainingArgs):
    """LABRAM training configuration."""
    max_epochs: int = 30

    weight_decay: float = 0.05
    weight_decay_end: Optional[float] = None
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_schedule: str = "cosine"  # 'cosine' or 'cycle'
    max_lr: float = 8e-4
    encoder_lr_scale: float = 0.1
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2  # For OneCycleLR
    min_lr: float = 8e-6

    # Layer-wise decay
    layer_decay: float = 0.9

    # Training options
    use_amp: bool = True
    freeze_encoder: bool = True
    label_smoothing: float = 0.1

    # Model EMA
    model_ema: bool = False
    model_ema_decay: float = 0.9999
    model_ema_force_cpu: bool = False


class LabramLoggingArgs(BaseLoggingArgs):
    """LABRAM logging configuration."""
    experiment_name: str = "labram"
    output_dir: str = "/path/to/your/code/baseline/labram/log"
    ckpt_dir: str = "/path/to/your/code/baseline/labram/ckpt"

    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"  # 'wandb', 'comet', or 'both'
    project: Optional[str] = "labram"
    entity: Optional[str] = None
    api_key: Optional[str] = None
    offline: bool = False

    tags: List[str] = Field(default_factory=lambda: [])

    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 5


class LabramConfig(AbstractConfig):
    """LABRAM configuration that extends AbstractConfig."""
    
    model_type: str = "labram"
    
    data: LabramDataArgs = Field(default_factory=LabramDataArgs)
    model: LabramModelArgs = Field(default_factory=LabramModelArgs)
    training: LabramTrainingArgs = Field(default_factory=LabramTrainingArgs)
    logging: LabramLoggingArgs = Field(default_factory=LabramLoggingArgs)

    def validate_config(self) -> bool:
        """Validate LABRAM-specific configuration."""
        # Check patch size requirements
        if self.model.patch_size <= 0:
            return False
            
        # Check if EEG size is divisible by patch size
        if self.model.eeg_size % self.model.patch_size != 0:
            return False
        
        # Check embed dimensions
        if self.model.embed_dim % self.model.num_heads != 0:
            return False
        
        # Check learning rate schedule
        if self.training.lr_schedule not in ["cosine", "linear"]:
            return False
            
        return True 