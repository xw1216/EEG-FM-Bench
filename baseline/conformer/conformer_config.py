"""
Conformer Configuration that inherits from AbstractConfig.
"""

from typing import Optional, List
from pydantic import Field

from baseline.abstract.classical import ClassicalDataArgs, ClassicalModelArgs, ClassicalTrainingArgs, ClassicalConfig, \
    ClassicalLoggingArgs


class ConformerLoggingArgs(ClassicalLoggingArgs):
    """Conformer logging configuration."""
    experiment_name: str = "eeg-conformer"
    output_dir: str = "/path/to/your/code/baseline/Conformer/log"
    ckpt_dir: str = "/path/to/your/code/baseline/Conformer/ckpt"

    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"  # 'wandb', 'comet', or 'both'
    project: Optional[str] = "eeg-conformer"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class ConformerConfig(ClassicalConfig):
    """Conformer configuration that extends AbstractConfig."""

    model_type: str = "eeg-conformer"

    data: ClassicalDataArgs = Field(default_factory=ClassicalDataArgs)
    model: ClassicalModelArgs = Field(default_factory=ClassicalModelArgs)
    training: ClassicalTrainingArgs = Field(default_factory=ClassicalTrainingArgs)
    logging: ConformerLoggingArgs = Field(default_factory=ConformerLoggingArgs)

    def validate_config(self) -> bool:
        """Validate Conformer-specific configuration."""
        # Check learning rate schedule
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False

        return True 