"""
Abstract trainer base class for baseline models.
"""
import datetime
import os
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import comet_ml
import datasets
import pandas as pd
import torch
import wandb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, cohen_kappa_score, f1_score
from torch import Tensor
from torch.utils.data import DataLoader

from baseline.abstract.adapter import AbstractDataLoaderFactory
from common.config import AbstractConfig
from baseline.utils.utils import seed_torch
from common.log import setup_log
from data.processor.wrapper import get_dataset_n_class, get_dataset_category
from common.distributed.env import get_is_master, get_global_rank, get_local_rank, get_world_size, get_master_addr, \
    get_master_port, get_specific_dirname
from common.distributed.loader import DistributedGroupBatchSampler
from common.utils import clean_torch_distributed

logger = logging.getLogger("baseline")


METRIC_PRECISION_DICT = {
    "lr": "6e",
    "header_lr": "6e",
    "encoder_lr": "6e",
    "gram": "2f",
    "accuracy": "3f",
    "acc": "3f",
    "f1": "3f",
    "pr": "3f",
    "recall": "3f",
    "cohen_kappa": "3f",
    "auroc": "3f",
    "auc_pr": "3f",
    "balanced_accuracy": "3f",
    "balanced_acc": "3f",
    "f1_weighted": "3f",
    "loss": "4f",
}


def format_console_log_dict(log_data: dict, prefix: str = 'train') -> str:
    """
    Format log dictionary with proper precision.

    Args:
        log_data: Dictionary of log metrics
        prefix: Prefix to remove from keys (e.g., 'train/')

    Returns:
        Formatted log string
    """
    prefix = f"{prefix}/"
    log_data = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in log_data.items()}
    formatted_log = ", ".join([
        f"{key}: {value:.{METRIC_PRECISION_DICT.get(key, '5e')}}" if isinstance(value, float)
        else f"{key}: {value}"
        for key, value in log_data.items()
    ])
    formatted_log = f"{prefix[:-1]} {formatted_log}"
    return formatted_log


class AbstractTrainer(ABC):
    """Abstract base trainer for all baseline models."""
    
    def __init__(self, cfg: AbstractConfig):
        self.cfg = cfg
        self.model_type = cfg.model_type
        self.multitask = cfg.multitask

        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_fn = None

        self.epoch = 0
        self.current_step = 0

        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        
        # Dataset information
        self.ds_conf = cfg.data.datasets
        self.num_ds = len(self.ds_conf)

        self.ds_info = {}
        self.montage_info = {}
        self.dataloader_factory: Optional[AbstractDataLoaderFactory] = None

        self.start_time = datetime.datetime.now()
        self.comet_experiment = None

    
    def setup_distributed(self):
        """Setup distributed training environment."""
        rank = get_global_rank()
        local_rank = get_local_rank()
        world_size = get_world_size()
        master_addr = get_master_addr()
        master_port = get_master_port(
            job_id=int(os.environ.get("SLURM_JOB_ID", -1)),
            port=self.cfg.master_port
        )

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(local_rank)

        assert 0 <= local_rank < 8
        torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
    
    def setup_logging(self):
        """Setup logging configuration."""
        if get_is_master():
            dirname = get_specific_dirname()
            output_dir = Path(self.cfg.logging.output_dir, dirname)
            output_dir.mkdir(parents=True, exist_ok=True)

            log_file = output_dir / f"{self.cfg.model_type}_trainer.log"
            setup_log(
                file_path=str(log_file),
                start_time=self.start_time.timestamp(),
                name="baseline",
                level="INFO"
            )

        logger.info(f"Starting {self.cfg.model_type} training with "
                   f"{self.num_ds} dataset(s): {list(self.ds_conf.keys())}")

    def init_cloud_logging(self):
        """Initialize cloud logging (wandb, comet, etc.)."""
        if not self.cfg.logging.use_cloud:
            return

        if get_is_master():
            # Initialize logging based on backend configuration
            backend = self.cfg.logging.cloud_backend.lower()

            if backend in ['wandb', 'both']:
                self._init_wandb()

            if backend in ['comet', 'both']:
                self._init_comet()

    def _init_wandb(self):
        """Initialize wandb logging."""
        try:
            # Create wandb metrics list
            wandb_metrics = []
            if self.multitask:
                wandb_metrics = ["train/step"]

            for ds_name in self.ds_conf.keys():
                if not self.multitask:
                    wandb_metrics.append(f"{ds_name}/train/step")
                wandb_metrics.extend([
                    f"{ds_name}/eval/epoch",
                    f"{ds_name}/test/epoch"
                ])

            # Setup wandb configuration with unified parameters
            wandb_config = {
                'project': self.cfg.logging.project or self.cfg.logging.experiment_name,
                'name': (
                    f"{self.model_type}_{'uni' if self.cfg.multitask else 'sep'}"
                    f"_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
                ),
                'config': self.cfg.model_dump(),
                'tags': self.cfg.logging.tags,
                'mode': 'offline' if self.cfg.logging.offline else 'online',
            }

            # Add optional parameters if specified
            if self.cfg.logging.entity:
                wandb_config['entity'] = self.cfg.logging.entity

            # Set API key if specified
            if self.cfg.logging.api_key:
                os.environ['WANDB_API_KEY'] = self.cfg.logging.api_key

            wandb.init(**wandb_config)

            # Define step metrics
            if self.multitask:
                wandb.define_metric("train/step")

            for metric in wandb_metrics:
                idx = metric.rfind('/')
                if idx == -1:
                    raise ValueError('No prefix to set metric')
                wandb.define_metric(metric)
                group = metric[:idx]
                wandb.define_metric(f'{group}/*', step_metric=metric)

            logger.info("Wandb logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")

    def _init_comet(self):
        try:
            # Setup comet configuration with unified parameters
            comet_config = {}

            # Set API key (from config or environment)
            api_key = self.cfg.logging.api_key or os.getenv('COMET_API_KEY')
            if not api_key:
                logger.warning("Comet API key not found, skipping comet logging")
                return

            comet_config['api_key'] = api_key
            comet_config['project_name'] = self.cfg.logging.project or self.cfg.logging.experiment_name

            if self.cfg.logging.entity:
                comet_config['workspace'] = self.cfg.logging.entity

            comet_config['experiment_name'] = (
                f"{self.model_type}_{'uni' if self.cfg.multitask else 'sep'}"
                f"_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
            )

            # Initialize comet experiment
            self.comet_experiment = comet_ml.Experiment(**comet_config)

            # Log configuration
            self.comet_experiment.log_parameters(self.cfg.model_dump())
            self.comet_experiment.add_tags(self.cfg.logging.tags)

            logger.info("Comet.ml logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize comet.ml: {e}")
            self.comet_experiment = None

    def finish_cloud_logging(self):
        """Finish cloud logging."""
        if not get_is_master():
            return

        backend = self.cfg.logging.cloud_backend.lower()

        if backend in ['wandb', 'both']:
            self._finish_wandb()

        if backend in ['comet', 'both']:
            self._finish_comet()

    def _finish_wandb(self):
        """Finish wandb logging."""
        try:
            wandb.finish()
            logger.info("Wandb logging finished")
        except Exception as e:
            logger.warning(f"Error finishing wandb: {e}")

    def _finish_comet(self):
        """Finish comet.ml logging."""
        try:
            self.comet_experiment.end()
            logger.info("Comet.ml logging finished")
        except Exception as e:
            logger.warning(f"Error finishing comet.ml: {e}")

    def _create_ft_cloud_log_data(self, log_data: dict, prefix: str, ds_metric: dict):
        # eval epoch metrics
        cloud_data = deepcopy(log_data)

        # Add raw confusion matrix data for cloud logging backends
        for ds_name in ds_metric.keys():
            matrix = ds_metric[ds_name]['cm'].cpu().numpy()
            labels = self.ds_info[ds_name]['category']

            # Store raw matrix and labels for both wandb and comet to handle
            cloud_data.update({f"{ds_name}/{prefix}/cm_matrix": matrix})
            cloud_data.update({f"{ds_name}/{prefix}/cm_labels": labels})

        return cloud_data

    def _log_to_cloud(self, log_data: dict):
        """Log data to configured cloud services."""
        backend = self.cfg.logging.cloud_backend.lower()

        if backend in ['wandb', 'both']:
            self._log_to_wandb(log_data)

        if backend in ['comet', 'both']:
            self._log_to_comet(log_data)

    def _log_to_wandb(self, log_data: dict):
        """Log data to wandb."""
        try:
            # Separate confusion matrix data from regular metrics
            wandb_data = {}
            cm_data = {}

            for key, value in log_data.items():
                if 'cm_matrix' in key or 'cm_labels' in key:
                    cm_data[key] = value
                else:
                    wandb_data[key] = value

            # Create wandb tables for confusion matrices
            for key, matrix in cm_data.items():
                if key.endswith('cm_matrix'):
                    base_key = key.replace('cm_matrix', '')
                    labels_key = base_key + 'cm_labels'
                    if labels_key in cm_data:
                        labels = cm_data[labels_key]
                        # Create wandb table
                        df = pd.DataFrame(matrix, columns=labels)
                        confusion_table = wandb.Table(dataframe=df)
                        wandb_data[f"{base_key}/cm"] = confusion_table

            # Log all data to wandb
            wandb.log(wandb_data)
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    def _log_to_comet(self, log_data: dict):
        """Log data to comet.ml."""
        if self.comet_experiment is None:
            return

        try:
            # Separate confusion matrix data from regular metrics
            metrics = {}
            cm_data = {}

            for key, value in log_data.items():
                if 'cm_matrix' in key or 'cm_labels' in key:
                    cm_data[key] = value
                else:
                    metrics[key] = value

            # Log regular metrics
            if metrics:
                self.comet_experiment.log_metrics(metrics)

            # Log confusion matrices
            for key, matrix in cm_data.items():
                if key.endswith('cm_matrix'):
                    base_key = key.replace('cm_matrix', '')
                    labels_key = base_key + 'cm_labels'
                    if labels_key in cm_data:
                        labels = cm_data[labels_key]
                        self.comet_experiment.log_confusion_matrix(
                            matrix=matrix,
                            labels=labels,
                            title=f"Confusion Matrix - {base_key.replace('/', '_')}"
                        )
        except Exception as e:
            logger.warning(f"Failed to log to comet.ml: {e}")

    def _calculate_metrics_for_dataset(
            self,
            labels: torch.Tensor,
            logits: torch.Tensor,
            ds_name: str,
            prefix: str,
            loss: float,
    ) -> Dict[str, float]:
        label_np = labels.numpy()
        pred_np = torch.argmax(logits, dim=-1).numpy()

        n_class = self.ds_info[ds_name]['n_class']

        metrics = {
            f'{ds_name}/{prefix}/epoch': self.epoch,
            f'{ds_name}/{prefix}/loss': loss,
        }

        # Basic accuracy
        # noinspection PyUnresolvedReferences
        accuracy = (pred_np == label_np).mean()
        metrics[f'{ds_name}/{prefix}/acc'] = float(accuracy)

        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(label_np, pred_np)
        metrics[f'{ds_name}/{prefix}/balanced_acc'] = float(balanced_acc)

        if n_class == 2:
            # Binary classification metrics
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()

            try:
                auroc = roc_auc_score(label_np, probs)
                metrics[f'{ds_name}/{prefix}/auroc'] = float(auroc)
            except ValueError as e:
                logger.warning(f'Error calculating AUROC for {ds_name} {prefix}: {e}')
                metrics[f'{ds_name}/{prefix}/auroc'] = 0.0

            try:
                auc_pr = average_precision_score(label_np, probs)
                metrics[f'{ds_name}/{prefix}/auc_pr'] = float(auc_pr)
            except ValueError as e:
                logger.warning(f'Error calculating AUC-PR for {ds_name} {prefix}: {e}')
                metrics[f'{ds_name}/{prefix}/auc_pr'] = 0.0
        else:
            # Multi-class classification metrics
            cohen_kappa = cohen_kappa_score(label_np, pred_np)
            metrics[f'{ds_name}/{prefix}/cohen_kappa'] = float(cohen_kappa)

            f1_weighted = f1_score(label_np, pred_np, average='weighted')
            metrics[f'{ds_name}/{prefix}/f1'] = float(f1_weighted)

        return metrics

    def collect_dataset_info(self, mixed: bool, ds_name: str = ''):
        """Collect information about datasets for model setup."""
        logger.info(f"Collecting dataset information for {'multitask' if self.multitask else 'per dataset'} ...")

        if mixed:
            self.ds_info = {}
            for dataset_name, dataset_config in self.ds_conf.items():
                self.ds_info[dataset_name] = {
                    'config': dataset_config,
                    'n_class': get_dataset_n_class(dataset_name, dataset_config),
                    'category': get_dataset_category(dataset_name, dataset_config)
                }
                logger.info(f"Dataset {dataset_name} - {dataset_config} for mixed set")
        else:
            ds_conf = self.ds_conf[ds_name]
            self.ds_info = {
                ds_name: {
                    'config': ds_conf,
                    'n_class': get_dataset_n_class(ds_name, ds_conf),
                    'category': get_dataset_category(ds_name, ds_conf)
                }}
            logger.info(f"Dataset {ds_name} - {ds_conf} only")

    def _gather_tensor(self, tensor: Tensor, max_length: int) -> Optional[list[Tensor]]:
        exist_mask = torch.tensor([tensor.shape[0]], dtype=torch.int32, device=self.device)
        mask_gather_list = [torch.zeros_like(exist_mask) for _ in range(self.world_size)] \
            if get_is_master() else None
        torch.distributed.gather(exist_mask, gather_list=mask_gather_list, dst=0)

        tensor_pad = torch.zeros([max_length, *(tensor.shape[1:])], dtype=tensor.dtype, device=tensor.device)
        tensor_pad[:tensor.shape[0]] = tensor
        gather_list = [torch.zeros_like(tensor_pad) for _ in range(self.world_size)] \
            if get_is_master() else None
        torch.distributed.gather(tensor_pad, gather_list=gather_list, dst=0)

        if get_is_master():
            for i in range(len(gather_list)):
                gather_list[i] = gather_list[i][:mask_gather_list[i]]

        return gather_list

    def _gather_result(self, logits: Tensor, targets: Tensor) -> tuple[Optional[Tensor], Optional[Tensor]]:
        logits_list = self._gather_tensor(logits, self.cfg.data.batch_size)
        target_list = self._gather_tensor(targets, self.cfg.data.batch_size)

        if get_is_master():
            all_logits = torch.cat(logits_list, dim=0)
            all_target = torch.cat(target_list, dim=0)
            return all_logits.cpu(), all_target.cpu()
        return None, None

    @staticmethod
    def _calc_confusion_matrix(pred: Tensor, target: Tensor, n_class: int) -> Tensor:
        pred, target = pred.long(), target.long()

        linear_indices = target * n_class + pred
        conf_matrix_flat = torch.bincount(linear_indices, minlength=n_class * n_class)
        conf_matrix = conf_matrix_flat.reshape(n_class, n_class)

        return conf_matrix

    def _clip_grad_norm_(self):
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
        return grad_norm.detach().cpu().item()

    def create_dataloader(self, split: datasets.NamedSplit = datasets.Split.TRAIN):
        logger.info("Creating main training dataloader...")
        mixed = (split == datasets.Split.TRAIN and self.cfg.multitask)

        dataloaders, samplers = self.dataloader_factory.create_dataloader(
            datasets_config=self.ds_conf,
            mixed=mixed,
            num_replicas=self.world_size,
            rank=self.local_rank,
            split=split,
        )

        return dataloaders, samplers

    def create_single_dataloader(self, ds_name: str, ds_config: str, split: datasets.NamedSplit = datasets.Split.TRAIN):
        logger.info("Creating single main training dataloader...")

        dataloader, sampler = self.dataloader_factory.create_dataloader(
            datasets_config={ds_name: ds_config},
            mixed=False,
            num_replicas=self.world_size,
            rank=self.local_rank,
            split=split,
        )

        dataloader = dataloader[0]
        sampler = sampler[0]

        return dataloader, sampler
    
    @abstractmethod
    def setup_model(self):
        """Setup model architecture."""
        pass

    def setup_optim_params(self, model):
        head_params = []
        encoder_params = []

        for name, param in model.named_parameters():
            if 'classifier' in name or 'conv_router' in name:
                head_params.append(param)
            else:
                encoder_params.append(param)

        params = [{'params': head_params, 'lr': self.cfg.training.max_lr}]

        # Add encoder parameters if not frozen
        if not self.cfg.training.freeze_encoder:
            encoder_lr = self.cfg.training.max_lr * self.cfg.training.encoder_lr_scale
            params.append({'params': encoder_params, 'lr': encoder_lr})
        else:
            # Freeze encoder parameters
            for param in encoder_params:
                param.requires_grad = False
            logger.info("Encoder parameters frozen")

        return params

    def setup_optimizer_and_scheduler(self, model, train_loader: DataLoader):
        params = self.setup_optim_params(model)

        optimizer = torch.optim.AdamW(
            params,
            weight_decay=self.cfg.training.weight_decay
        )

        # Gradient scaler for mixed precision
        scaler = torch.amp.GradScaler(enabled=self.cfg.training.use_amp)

        # Learning rate scheduler
        warmup_steps = len(train_loader) * self.cfg.training.warmup_epochs
        total_steps = len(train_loader) * self.cfg.training.max_epochs

        if self.cfg.training.lr_schedule == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[p['lr'] for p in params],
                total_steps=total_steps,
                pct_start=self.cfg.training.pct_start
            )
        elif self.cfg.training.lr_schedule == 'cosine':  # warm cosine annealing
            warm_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.cfg.training.warmup_scale,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.cfg.training.min_lr
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warm_scheduler, cos_scheduler],
                milestones=[warmup_steps]
            )
        else:
            raise NotImplementedError('Unknown learning rate schedule')

        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler

    def train_step(self, batch, labels):
        with torch.amp.autocast('cuda', enabled=self.cfg.training.use_amp, dtype=torch.bfloat16):
            logits = self.model(batch)

        loss = self.loss_fn(logits, labels)
        return logits, loss

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
            # for name, param in self.model.named_parameters():
            #     if get_is_master() and param.grad is not None:
            #         logger.info(
            #             f"{name} "
            #             f"Range: [{param.grad.min():.8f}, {param.grad.max():.8f}], "
            #             f"Scale: {param.grad.abs().mean():.8f}")

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
                        'train/header_lr': self.scheduler.get_last_lr()[0],
                    }

                    if not self.cfg.training.freeze_encoder:
                        log_data['train/encoder_lr'] = self.scheduler.get_last_lr()[1]

                    if not self.multitask:
                        log_data = {f"{ds_name}/{key}": value for key, value in log_data.items()}

                    # Log to cloud services
                    if self.cfg.logging.use_cloud:
                        self._log_to_cloud(log_data)

                    logger.info(format_console_log_dict(log_data, prefix='train'))

            self.current_step += 1
            self.scheduler.step()
    

    def eval_step(self, batch, labels):
        with torch.amp.autocast('cuda', enabled=self.cfg.training.use_amp, dtype=torch.bfloat16):
            logits = self.model(batch)

        loss = self.loss_fn(logits, labels)
        return logits, loss

    def eval_epoch(self, dataloaders: list[DataLoader], prefix: str):
        """Evaluate one epoch and return metrics."""
        if get_is_master():
            logger.info(f"Starting {prefix} evaluation...")

        self.model.eval()

        overall_metrics = {}
        for ds_name in self.ds_info.keys():
            n_class = self.ds_info[ds_name]['n_class']
            overall_metrics[ds_name] = {
                'loss_sum': torch.zeros([1], dtype=torch.float64, device=self.device),
                'cm': torch.zeros((n_class, n_class), dtype=torch.int64, device=self.device),
                'cnt': torch.zeros(1, dtype=torch.int64, device=self.device),
                'logits': [],
                'labels': [],
            }

        with torch.no_grad():
            for dataloader in dataloaders:
                for batch in dataloader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    labels = batch['label']
                    ds_name = batch['montage'][0].split('/')[0]
                    n_class = self.ds_info[ds_name]['n_class']

                    # Forward pass with mixed precision
                    logits, loss = self.train_step(batch, labels)

                    logits = logits.float()
                    pred = torch.argmax(logits, dim=1).detach()
                    cm = self._calc_confusion_matrix(pred, labels.detach(), n_class)

                    overall_metrics[ds_name]['loss_sum'] += loss.detach() * len(batch)
                    overall_metrics[ds_name]['cnt'] += len(batch)
                    overall_metrics[ds_name]['cm'] += cm.detach()

                    logits_across, labels_across = self._gather_result(logits.detach(), labels.detach())
                    if get_is_master():
                        overall_metrics[ds_name]['logits'].append(logits_across.cpu())
                        overall_metrics[ds_name]['labels'].append(labels_across.cpu())

                torch.distributed.barrier()

            log_dict = {}
            for ds_name in self.ds_info.keys():
                torch.distributed.all_reduce(overall_metrics[ds_name]['loss_sum'], op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(overall_metrics[ds_name]['cnt'], op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(overall_metrics[ds_name]['cm'], op=torch.distributed.ReduceOp.SUM)

                overall_metrics[ds_name]['loss'] = overall_metrics[ds_name]['loss_sum'] / overall_metrics[ds_name][
                    'cnt'].float()

                # Calculate metrics on aggregated data (only master process in distributed mode)
                if get_is_master():
                    labels_all = torch.concat(overall_metrics[ds_name]['labels'], dim=0)
                    logits_all = torch.concat(overall_metrics[ds_name]['logits'], dim=0)
                    loss_metric = overall_metrics[ds_name]['loss'].detach().cpu().item()
                    metrics = self._calculate_metrics_for_dataset(
                        labels=labels_all,
                        logits=logits_all,
                        ds_name=ds_name,
                        prefix=prefix,
                        loss=loss_metric
                    )

                    log_dict = log_dict | metrics
                    log_console = format_console_log_dict(metrics, prefix=f"{ds_name}/{prefix}")
                    logger.info(log_console)

            if get_is_master() and self.cfg.logging.use_cloud:
                log_cloud = self._create_ft_cloud_log_data(log_dict, prefix, overall_metrics)
                self._log_to_cloud(log_cloud)

            torch.distributed.barrier()

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        pass
    
    def save_checkpoint(self, ds_name: Optional[str] = None, is_milestone: bool = False, **kwargs):
        if not get_is_master():
            return

        if ds_name is None:
            ds_name = 'unified'
            checkpoint_dir = Path(self.cfg.logging.ckpt_dir, ds_name)
        else:
            checkpoint_dir = Path(self.cfg.logging.ckpt_dir, 'seperated', ds_name)


        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.cfg.model_dump(),
            'dataset_name': ds_name,
        }

        # Save checkpoint
        suffix = 'last' if is_milestone else f'epoch_{self.epoch}'
        checkpoint_path = checkpoint_dir / f'{self.model_type}_{ds_name}_{suffix}.pt'
        torch.save(checkpoint, checkpoint_path)

        logger.info(f"Checkpoint saved: {ds_name}: {checkpoint_path}")

    def run(self):
        seed_torch(self.cfg.seed)
        self.setup_distributed()
        self.setup_logging()
        self.init_cloud_logging()

        logger.info(f"Starting {self.cfg.model_type} training with configuration:")
        logger.info(f"  - Datasets: {self.num_ds} {list(self.cfg.data.datasets.keys())}")
        logger.info(f"  - Multitask: {self.cfg.multitask}")
        logger.info(f"  - Max epochs: {self.cfg.training.max_epochs}")
        logger.info(f"  - Output directory: {self.cfg.logging.output_dir}")

        """Main training loop - supports both multitask and separate models patterns."""
        if self.cfg.multitask:
            logger.info("Using separate models training pattern - one model per dataset")
            self.run_unified_training()
        else:
            logger.info("Using unified/multitask training pattern - single shared model")
            self.run_separate_training()

    def run_unified_training(self):
        """Original unified training loop for multitask or single dataset training."""
        torch.distributed.barrier()

        self.collect_dataset_info(mixed=True)
        model = self.setup_model()

        train_loader, train_sampler = self.create_dataloader(datasets.Split.TRAIN)
        valid_loaders, _ = self.create_dataloader(datasets.Split.VALIDATION)
        test_loaders, _ = self.create_dataloader(datasets.Split.TEST)

        if not isinstance(train_loader, DataLoader) or not isinstance(train_sampler, DistributedGroupBatchSampler):
            raise TypeError('train_loader and train_sampler must be of type DataLoader')

        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(model, train_loader)

        logger.info(f"Training setup complete. Starting {self.cfg.training.max_epochs} epochs...")

        # Training loop
        for epoch in range(self.cfg.training.max_epochs):
            self.epoch = epoch

            torch.distributed.barrier()

            self.train_epoch(train_loader, train_sampler)

            self.eval_epoch(valid_loaders, 'eval')
            self.eval_epoch(test_loaders, 'test')

            # Save checkpoint
            if (epoch + 1) % self.cfg.logging.ckpt_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint(is_milestone=True)

        self.finish_cloud_logging()
        clean_torch_distributed(self.local_rank)

        logger.info("Training completed successfully!")

    def run_separate_training(self):
        """Main training loop for separate models pattern - train one model per dataset."""
        torch.distributed.barrier()

        logger.info(f"Starting separate models training for {self.num_ds} datasets")

        # Train each dataset separately
        for i, (ds_name, ds_config) in enumerate(self.ds_conf.items()):
            if get_is_master():
                logger.info(f"Training dataset {i + 1}/{self.num_ds}: {ds_name}")

            self.collect_dataset_info(mixed=False, ds_name=ds_name)
            model = self.setup_model()

            train_loader, train_sampler = self.create_single_dataloader(ds_name, ds_config, datasets.Split.TRAIN)
            valid_loader, _ = self.create_single_dataloader(ds_name, ds_config, datasets.Split.VALIDATION)
            test_loader, _ = self.create_single_dataloader(ds_name, ds_config, datasets.Split.TEST)

            if not isinstance(train_loader, DataLoader) or not isinstance(train_sampler, DistributedGroupBatchSampler):
                raise TypeError('train_loader and train_sampler must be of type DataLoader')
            if not isinstance(valid_loader, DataLoader):
                raise TypeError('valid_loader must be of type DataLoader')
            if not isinstance(test_loader, DataLoader):
                raise TypeError('test_loader must be of type DataLoader')

            # Setup optimizer and scheduler
            self.setup_optimizer_and_scheduler(model, train_loader)

            logger.info(f"Per dataset training setup complete for {ds_name}. ")
            logger.info(f"Starting {self.cfg.training.max_epochs} epochs...")

            # Training loop for this dataset
            for epoch in range(self.cfg.training.max_epochs):
                self.epoch = epoch

                torch.distributed.barrier()

                self.train_epoch(train_loader, train_sampler)

                self.eval_epoch([valid_loader], 'eval')
                self.eval_epoch([test_loader], 'test')

                # Save checkpoint
                if (epoch + 1) % self.cfg.logging.ckpt_interval == 0:
                    self.save_checkpoint(ds_name=ds_name)

            self.save_checkpoint(ds_name, is_milestone=True)

            logger.info(f"Training completed for {ds_name}!")

            self.epoch = 0
            self.current_step = 0

        self.finish_cloud_logging()
        clean_torch_distributed(self.local_rank)
        logger.info("Separate models training completed for all datasets!")

