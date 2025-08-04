#!/usr/bin/env python3
import sys
from pathlib import Path

from omegaconf import OmegaConf

from baseline.abstract.factory import ModelRegistry
from common.path import get_conf_file_path
from common.utils import setup_yaml


def main():
    """Main training function that can handle any registered baseline model."""
    setup_yaml()
    
    # Parse CLI arguments
    cli_args = OmegaConf.from_cli()

    if 'conf_file' not in cli_args:
        raise ValueError("Please provide a config file: conf_file=path/to/config.yaml")
    
    # Get model type from CLI args or config
    model_type: str = cli_args.get('model_type', None)

    # Load config file
    conf_file_path = get_conf_file_path(cli_args.conf_file)
    file_cfg = OmegaConf.load(conf_file_path)

    if model_type is None:
        model_type = file_cfg.get('model_type')

    # Validate model type
    available_models = ModelRegistry.list_models()
    if model_type not in available_models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
    
    # Create base config for the specified model type
    config_class = ModelRegistry.get_config_class(model_type)
    code_cfg = OmegaConf.create(config_class().model_dump())
    
    # Merge configurations: code defaults < file config < CLI args
    merged_config = OmegaConf.merge(code_cfg, file_cfg, cli_args)
    
    # Ensure model_type is set correctly
    merged_config.model_type = model_type
    
    # Convert to config object
    cfg_dict = OmegaConf.to_container(merged_config, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg_dict)
    
    # Validate configuration
    if not cfg.validate_config():
        raise ValueError(f"Invalid configuration for model type: {model_type}")
    
    # Setup output directory
    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and run trainer
    trainer = ModelRegistry.create_trainer(cfg)
    trainer.run()


def list_available_models():
    """List all available model types."""
    print("Available baseline models:")
    for model_type in ModelRegistry.list_models():
        print(f"  - {model_type}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        list_available_models()
    else:
        main() 