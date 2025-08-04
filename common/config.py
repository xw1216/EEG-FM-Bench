"""
Abstract configuration base class for baseline models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from pydantic import BaseModel, Field

class PreprocArgs(BaseModel):
    clean_middle_cache: bool = False
    num_preproc_arrow_writers: int = 4
    num_preproc_mid_workers: int = 6
    pretrain_datasets: list[str] = Field(default_factory=lambda: [])
    finetune_datasets: dict[str, str] = Field(default_factory=lambda: {})


class BaseDataArgs(BaseModel):
    """Base data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class BaseModelArgs(BaseModel):
    """Base model configuration."""
    pretrained_path: Optional[str] = None

    grad_cam: bool = False
    t_sne: bool = False
    grad_cam_target: str = 'channel'

class BaseTrainingArgs(BaseModel):
    """Base training configuration."""
    max_epochs: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    lr_schedule: str = "onecycle"  # 'onecycle' or 'cosine'
    max_lr: float = 1e-4
    encoder_lr_scale: float = 0.1
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2  # For OneCycleLR
    min_lr: float = 1e-6  # For CosineAnnealingLR

    use_amp: bool = True
    freeze_encoder: bool = True


class BaseLoggingArgs(BaseModel):
    """Base logging configuration."""
    experiment_name: str = "baseline"
    output_dir: str = "./logs"
    ckpt_dir: str = "./checkpoints"

    use_cloud: bool = False
    cloud_backend: str = "wandb"
    project: Optional[str] = None
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    log_step_interval: int = 1
    ckpt_interval: int = 1


class AbstractConfig(BaseModel, ABC):
    """Abstract base configuration class for all baseline models."""
    
    seed: int = 42
    master_port: int = 41001
    multitask: bool = False
    model_type: str = "base"  # To identify which model is being used
    conf_file: Optional[str] = None
    
    data: BaseDataArgs = Field(default_factory=BaseDataArgs)
    model: BaseModelArgs = Field(default_factory=BaseModelArgs)
    training: BaseTrainingArgs = Field(default_factory=BaseTrainingArgs)
    logging: BaseLoggingArgs = Field(default_factory=BaseLoggingArgs)

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate model-specific configuration requirements."""
        pass
