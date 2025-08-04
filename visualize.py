#!/usr/bin/env python3

import sys
import logging
from pathlib import Path

from omegaconf import OmegaConf

from baseline.utils.utils import seed_torch
from common.log import setup_log
from common.path import get_conf_file_path
from common.utils import setup_yaml
from common.config import AbstractConfig
from baseline.abstract.factory import ModelRegistry
from plot.conf import load_vis_conf_dict
from plot.baseline_visualizer import BaselineVisualizer


logger = logging.getLogger()


def detect_model_type(config_path: str) -> str:
    config_path = get_conf_file_path(config_path)
    file_cfg = OmegaConf.load(config_path)

    model_type = file_cfg.model_type
    if model_type in ModelRegistry.list_models():
        return 'baseline'
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def load_model_config(config_path: str) -> tuple[str, AbstractConfig]:
    model_type = detect_model_type(config_path)
    config_path = get_conf_file_path(config_path)

    file_cfg = OmegaConf.load(config_path)
    specific_model_type = file_cfg.get('model_type', 'eegpt')

    config_class = ModelRegistry.get_config_class(specific_model_type)
    code_cfg = OmegaConf.create(config_class().model_dump())

    cfg = OmegaConf.merge(code_cfg, file_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg)

    logger.info(f'change batch_size forcefully to 1')
    cfg.data.batch_size = 1

    logger.info(f'change pretrained model path to none')
    cfg.model.pretrained_path = None

    return model_type, cfg


def create_visualizer(model_config: AbstractConfig, vis_args):
    if isinstance(model_config, AbstractConfig):
        return BaselineVisualizer(model_config, vis_args)
    else:
        raise ValueError(f"Unsupported model config type: {type(model_config)}")


def main():
    """Main visualization function."""
    if len(sys.argv) != 4:
        print("Usage: python visualize.py <vis_type> <model_config> <vis_config>")
        print("  vis_type: 't_sne', 'grad_cam', or 'integrated_gradients'")
        print("  model_config: path to model configuration file")
        print("  vis_config: path to visualization configuration file")
        print("Supported model types:")
        print(f"  - baseline: {', '.join(ModelRegistry.list_models())}")
        sys.exit(1)
    
    vis_type = sys.argv[1]
    model_config_path = sys.argv[2]
    vis_config_path = sys.argv[3]
    
    # Setup logging
    setup_log()
    setup_yaml()

    logger.info(f"Starting {vis_type} visualization")
    logger.info(f"Model config: {model_config_path}")
    logger.info(f"Visualization config: {vis_config_path}")

    if not Path(vis_config_path).exists():
        raise FileNotFoundError(f"Visualization config file not found: {vis_config_path}")

    # Load model configuration
    model_type, model_config = load_model_config(model_config_path)
    logger.info(f"Loaded {type(model_config).__name__} configuration")

    # Import and run appropriate visualizer
    if vis_type == 't_sne':
        vis_config = load_vis_conf_dict(vis_config_path, vis_type)
        model_config.model.t_sne = True
    elif vis_type == 'grad_cam':
        vis_config = load_vis_conf_dict(vis_config_path, vis_type)
        model_config.model.grad_cam = True
        model_config.model.grad_cam_target = vis_config.grad_cam_target
    elif vis_type == 'integrated_gradients':
        vis_config = load_vis_conf_dict(vis_config_path, vis_type)
    else:
        raise ValueError(f"Unsupported visualization type: {vis_type}. Use 't_sne', 'grad_cam', or 'integrated_gradients'")

    logger.info(f'visualization config {vis_config}')
    logger.info(f'target model config {model_config}')

    if model_type == 'baseline':
        model_config.data.datasets = vis_config.datasets
    else:
        model_config.dist.seed = vis_config.seed

    seed_torch(vis_config.seed)
    visualizer = create_visualizer(model_config, vis_config)
    visualizer.run()

    logger.info(f"{vis_type} visualization completed successfully")



if __name__ == "__main__":
    main() 