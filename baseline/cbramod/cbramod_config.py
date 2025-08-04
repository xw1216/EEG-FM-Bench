"""
CBraMod Configuration that inherits from AbstractConfig.
"""

from typing import Dict, Optional, List
from pydantic import Field

from common.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class CBraModDataArgs(BaseDataArgs):
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class CBraModModelArgs(BaseModelArgs):
    # Pretrained model path
    pretrained_path: Optional[str] = None

    # CBraMod architecture parameters
    in_dim: int = 200
    out_dim: int = 200
    d_model: int = 200
    dim_ffn: int = 800
    n_layer: int = 12
    n_head: int = 8
    
    # Regularization
    dropout_rate: float = 0.1
    
    # Classification head
    head_dropout: float = 0.1
    mlp_hidden_dim: list[int] = Field(default_factory=lambda: [128])


class CBraModTrainingArgs(BaseTrainingArgs):
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
    freeze_encoder: bool = False


class CBraModLoggingArgs(BaseLoggingArgs):
    experiment_name: str = "cbramod"
    output_dir: str = "/path/to/your/code/baseline/cbramod/log"
    ckpt_dir: str = "/path/to/your/code/baseline/cbramod/ckpt"

    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"  # 'wandb', 'comet', or 'both'
    project: Optional[str] = "cbramod"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class CBraModConfig(AbstractConfig):
    model_type: str = "cbramod"
    
    data: CBraModDataArgs = Field(default_factory=CBraModDataArgs)
    model: CBraModModelArgs = Field(default_factory=CBraModModelArgs)
    training: CBraModTrainingArgs = Field(default_factory=CBraModTrainingArgs)
    logging: CBraModLoggingArgs = Field(default_factory=CBraModLoggingArgs)

    def validate_config(self) -> bool:
        """Validate CBraMod-specific configuration."""
        # Check model dimensions
        if self.model.d_model <= 0 or self.model.dim_ffn <= 0:
            return False
            
        # Check attention heads configuration
        if self.model.d_model % self.model.n_head != 0:
            return False
            
        # Check learning rate schedule
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False
            
        return True 