"""
EegNet Configuration that inherits from AbstractConfig.
"""

from typing import Optional, List
from pydantic import Field

from baseline.abstract.classical import ClassicalLoggingArgs, ClassicalConfig, ClassicalDataArgs, ClassicalModelArgs, \
    ClassicalTrainingArgs


class EegNetLoggingArgs(ClassicalLoggingArgs):
    """EegNet logging configuration."""
    experiment_name: str = "eegnet"
    output_dir: str = "/path/to/your/code/baseline/eegnet/log"
    ckpt_dir: str = "/path/to/your/code/baseline/eegnet/ckpt"

    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"  # 'wandb', 'comet', or 'both'
    project: Optional[str] = "eegnet"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class EegNetConfig(ClassicalConfig):
    """EegNet configuration that extends AbstractConfig."""

    model_type: str = "eegnet"

    data: ClassicalDataArgs = Field(default_factory=ClassicalDataArgs)
    model: ClassicalModelArgs = Field(default_factory=ClassicalModelArgs)
    training: ClassicalTrainingArgs = Field(default_factory=ClassicalTrainingArgs)
    logging: EegNetLoggingArgs = Field(default_factory=EegNetLoggingArgs)

    def validate_config(self) -> bool:
        """Validate EegNet-specific configuration."""
        # Check learning rate schedule
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False

        return True 