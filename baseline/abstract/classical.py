"""
Classical Configuration that inherits from AbstractConfig.
"""
import logging
from abc import ABC
from typing import Dict, Optional, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from pydantic import Field

from common.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs
from baseline.abstract.trainer import AbstractTrainer, format_console_log_dict
from common.distributed.env import get_is_master
from common.distributed.loader import DistributedGroupBatchSampler
from data.processor.wrapper import get_dataset_montage, get_dataset_n_class, get_dataset_category, get_dataset_patch_len

logger = logging.getLogger('baseline')


class ClassicalDataArgs(BaseDataArgs):
    """Classical data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class ClassicalModelArgs(BaseModelArgs):
    """Classical model configuration."""
    # Pretrained model path
    pretrained_path: Optional[str] = None


class ClassicalTrainingArgs(BaseTrainingArgs):
    """Classical training configuration."""
    max_epochs: int = 100

    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_schedule: str = "cosine"  # 'onecycle' or 'cosine'
    max_lr: float = 2e-4
    encoder_lr_scale: float = 1.0
    warmup_epochs: int = 5
    warmup_scale: float = 1e-1
    pct_start: float = 0.1  # For OneCycleLR
    min_lr: float = 2e-5  # For CosineAnnealingLR

    use_amp: bool = False


class ClassicalLoggingArgs(BaseLoggingArgs):
    """Classical logging configuration."""
    experiment_name: str = "classical"
    output_dir: str = "/path/to/your/code/baseline/classical/log"
    ckpt_dir: str = "/path/to/your/code/baseline/classical/ckpt"

    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"  # 'wandb', 'comet', or 'both'
    project: Optional[str] = "classical"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class ClassicalConfig(AbstractConfig):
    """Classical configuration that extends AbstractConfig."""

    model_type: str = "classical"

    data: ClassicalDataArgs = Field(default_factory=ClassicalDataArgs)
    model: ClassicalModelArgs = Field(default_factory=ClassicalModelArgs)
    training: ClassicalTrainingArgs = Field(default_factory=ClassicalTrainingArgs)
    logging: ClassicalLoggingArgs = Field(default_factory=ClassicalLoggingArgs)

    def validate_config(self) -> bool:
        """Validate Classical-specific configuration."""
        # Check learning rate schedule
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False

        return True

class ClassicalTrainer(AbstractTrainer, ABC):
    def __init__(self, cfg: ClassicalConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.sfreq = 256

        self.encoder = None
        self.loss_fn = nn.CrossEntropyLoss()

    def load_checkpoint(self, checkpoint_path: str):
        pass

    def collect_dataset_info(self, mixed: bool, ds_name: str = ''):
        logger.info(f"Collecting dataset information for {'multitask' if self.multitask else 'per dataset'} ...")
        if mixed:
            raise NotImplementedError(f'{self.cfg.model_type} does not support mixed datasets.')

        ds_conf = self.ds_conf[ds_name]
        montages: dict = get_dataset_montage(ds_name, ds_conf)
        if len(montages) == 0 or len(montages) > 1:
            raise ValueError(f'{self.cfg.model_type} does not support dataset with multiple montages.')
        else:
            montage = next(iter(montages.values()))

        self.ds_info = {
            ds_name: {
                'config': ds_conf,
                'n_class': get_dataset_n_class(ds_name, ds_conf),
                'n_ch': len(montage),
                'category': get_dataset_category(ds_name, ds_conf),
                'wnd_sec': get_dataset_patch_len(ds_name, ds_conf),
            }}
        logger.info(f"Dataset {ds_name} - {ds_conf} only")

    def setup_optim_params(self, model):
        encoder_params = []

        for name, param in model.named_parameters():
                encoder_params.append(param)

        params = [{'params': encoder_params, 'lr': self.cfg.training.max_lr}]

        return params

    def run_unified_training(self):
        raise NotImplementedError(f'{self.cfg.model_type} does not support unified training.')

    def train_epoch(self, train_loader: DataLoader, train_sampler: DistributedGroupBatchSampler):
        self.model.train()
        train_sampler.set_epoch(self.epoch)

        batch: dict
        for step_in_epoch, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch['label']
            ds_name = batch['montage'][0].split('/')[0]

            # Forward pass with mixed precision
            logits, loss = self.train_step(batch, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at step {self.current_step}")

            # Backward pass
            self.scaler.scale(loss).backward()

            grad_norm = self._clip_grad_norm_()

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Logging with distributed reduction
            if self.current_step % self.cfg.logging.log_step_interval == 0:
                # Calculate step accuracy
                preds = torch.argmax(logits, dim=-1)
                step_acc = (preds == labels).float().mean()

                # Create tensors for distributed reduction
                loss_tensor = loss.clone().detach()
                acc_tensor = step_acc.clone().detach()

                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(acc_tensor, op=torch.distributed.ReduceOp.AVG)

                if get_is_master():
                    log_data = {
                        'train/epoch': self.epoch,
                        'train/step': self.current_step,
                        'train/loss_ce': loss_tensor.cpu().item(),
                        'train/acc': acc_tensor.cpu().item(),
                        'train/grad_norm': grad_norm,
                        'train/encoder_lr': self.scheduler.get_last_lr()[0],
                    }

                    if not self.multitask:
                        log_data = {f"{ds_name}/{key}": value for key, value in log_data.items()}

                    # Log to cloud services
                    if self.cfg.logging.use_cloud:
                        self._log_to_cloud(log_data)

                    logger.info(format_console_log_dict(log_data, prefix='train'))

            self.current_step += 1
            self.scheduler.step()