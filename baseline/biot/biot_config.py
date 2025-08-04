"""
BIOT Configuration that inherits from AbstractConfig.
"""

from typing import Dict, Optional, List
from pydantic import Field

from common.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class BiotDataArgs(BaseDataArgs):
    """BIOT data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class BiotModelArgs(BaseModelArgs):
    """BIOT model configuration."""
    # Pretrained model path
    pretrained_path: Optional[str] = '/path/to/your/code/biot/ckpt/biot-EEG-six-datasets-18-channels.ckpt'

    # BIOT architecture parameters
    emb_size: int = 256
    heads: int = 8
    depth: int = 4
    max_channels: int = 18
    
    # STFT parameters
    n_fft: int = 200
    hop_length: int = 100
    
    # Channel adaptation
    use_channel_conv: bool = True
    
    # Classification head
    head_dropout: float = 0.1
    mlp_hidden_dim: list[int] = Field(default_factory=lambda: [128])


class BiotTrainingArgs(BaseTrainingArgs):
    """BIOT training configuration."""
    max_epochs: int = 50

    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_schedule: str = "cosine"  # 'onecycle' or 'cosine'
    max_lr: float = 4e-4
    encoder_lr_scale: float = 0.1
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2  # For OneCycleLR
    min_lr: float = 4e-6  # For CosineAnnealingLR

    use_amp: bool = False
    freeze_encoder: bool = True


class BiotLoggingArgs(BaseLoggingArgs):
    """BIOT logging configuration."""
    experiment_name: str = "biot"
    output_dir: str = "/path/to/your/code/biot/log"
    ckpt_dir: str = "/path/to/your/code/biot/ckpt"

    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"  # 'wandb', 'comet', or 'both'
    project: Optional[str] = "biot"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class BiotConfig(AbstractConfig):
    """BIOT configuration that extends AbstractConfig."""
    
    model_type: str = "biot"
    
    data: BiotDataArgs = Field(default_factory=BiotDataArgs)
    model: BiotModelArgs = Field(default_factory=BiotModelArgs)
    training: BiotTrainingArgs = Field(default_factory=BiotTrainingArgs)
    logging: BiotLoggingArgs = Field(default_factory=BiotLoggingArgs)

    def validate_config(self) -> bool:
        """Validate BIOT-specific configuration."""
        # Check STFT parameters
        if self.model.n_fft <= 0 or self.model.hop_length <= 0:
            return False
            
        # Check embed dimensions
        if self.model.emb_size <= 0:
            return False
            
        # Check heads configuration
        if self.model.emb_size % self.model.heads != 0:
            return False
        
        # Check learning rate schedule
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False
            
        return True 