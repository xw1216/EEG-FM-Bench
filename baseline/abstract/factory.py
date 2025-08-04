"""
Factory classes for creating baseline models, adapters, and trainers.
"""

from typing import Type, Dict, Any, Optional
from common.config import AbstractConfig
from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.trainer import AbstractTrainer


class ModelRegistry:
    """Registry for baseline models."""
    
    configs: Dict[str, Type[AbstractConfig]] = {}
    adapters: Dict[str, Optional[Type[AbstractDataLoaderFactory]]] = {}
    trainers: Dict[str, Type[AbstractTrainer]] = {}
    
    @classmethod
    def register_model(
        cls,
        model_type: str,
        config_class: Type[AbstractConfig],
        adapter_class: Optional[Type[AbstractDataLoaderFactory]],
        trainer_class: Type[AbstractTrainer]
    ):
        """Register a new model type."""
        cls.configs[model_type] = config_class
        cls.adapters[model_type] = adapter_class
        cls.trainers[model_type] = trainer_class
    
    @classmethod
    def get_config_class(cls, model_type: str) -> Type[AbstractConfig]:
        """Get configuration class for model type."""
        if model_type not in cls.configs:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(cls.configs.keys())}")
        return cls.configs[model_type]
    
    @classmethod
    def get_adapter_class(cls, model_type: str) -> Type[AbstractDataLoaderFactory]:
        """Get adapter class for model type."""
        if model_type not in cls.adapters:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(cls.adapters.keys())}")
        return cls.adapters[model_type]
    
    @classmethod
    def get_trainer_class(cls, model_type: str) -> Type[AbstractTrainer]:
        """Get trainer class for model type."""
        if model_type not in cls.trainers:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(cls.trainers.keys())}")
        return cls.trainers[model_type]
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model types."""
        return list(cls.configs.keys())
    
    @classmethod
    def create_config(cls, model_type: str, **kwargs) -> AbstractConfig:
        """Create configuration instance for model type."""
        config_class = cls.get_config_class(model_type)
        return config_class(model_type=model_type, **kwargs)
    
    @classmethod
    def create_trainer(cls, config: AbstractConfig) -> AbstractTrainer:
        """Create trainer instance for configuration."""
        trainer_class = cls.get_trainer_class(config.model_type)
        return trainer_class(config)


class BaselineModelFactory:
    """Factory for creating baseline model components."""
    
    @staticmethod
    def create_from_config(config_dict: Dict[str, Any]) -> AbstractTrainer:
        """Create trainer from configuration dictionary."""
        model_type = config_dict.get('model_type', 'eegpt')
        
        # Create configuration
        config = ModelRegistry.create_config(model_type, **config_dict)
        
        # Validate configuration
        if not config.validate_config():
            raise ValueError(f"Invalid configuration for model type: {model_type}")
        
        # Create trainer
        trainer = ModelRegistry.create_trainer(config)
        
        return trainer
    
    @staticmethod
    def list_available_models() -> list[str]:
        """List all available model types."""
        return ModelRegistry.list_models()



