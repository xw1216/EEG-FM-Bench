from baseline.abstract.factory import ModelRegistry
from baseline.cbramod.cbramod_adapter import CBraModDataLoaderFactory
from baseline.cbramod.cbramod_config import CBraModConfig
from baseline.cbramod.cbramod_trainer import CBraModTrainer
from baseline.conformer.conformer_config import ConformerConfig
from baseline.conformer.conformer_trainer import ConformerTrainer
from baseline.eegnet.eegnet_config import EegNetConfig
from baseline.eegnet.eegnet_trainer import EegNetTrainer
from baseline.eegpt.eegpt_adapter import EegptDataLoaderFactory
from baseline.eegpt.eegpt_config import EegptConfig
from baseline.eegpt.eegpt_trainer import EegptTrainer
from baseline.labram.labram_adapter import LabramDataLoaderFactory
from baseline.labram.labram_config import LabramConfig
from baseline.labram.labram_trainer import LabramTrainer
from baseline.bendr.bendr_config import BendrConfig
from baseline.bendr.bendr_trainer import BendrTrainer
from baseline.biot.biot_config import BiotConfig
from baseline.biot.biot_trainer import BiotTrainer


ModelRegistry.register_model(
    model_type='eegpt',
    config_class=EegptConfig,
    adapter_class=EegptDataLoaderFactory,
    trainer_class=EegptTrainer
)

ModelRegistry.register_model(
    model_type='labram',
    config_class=LabramConfig,
    adapter_class=LabramDataLoaderFactory,
    trainer_class=LabramTrainer
)

ModelRegistry.register_model(
    model_type='bendr',
    config_class=BendrConfig,
    adapter_class=None,
    trainer_class=BendrTrainer
)

ModelRegistry.register_model(
    model_type='biot',
    config_class=BiotConfig,
    adapter_class=None,
    trainer_class=BiotTrainer
)

ModelRegistry.register_model(
    model_type='cbramod',
    config_class=CBraModConfig,
    adapter_class=CBraModDataLoaderFactory,
    trainer_class=CBraModTrainer
)

ModelRegistry.register_model(
    model_type='eegnet',
    config_class=EegNetConfig,
    adapter_class=None,
    trainer_class=EegNetTrainer
)

ModelRegistry.register_model(
    model_type='conformer',
    config_class=ConformerConfig,
    adapter_class=None,
    trainer_class=ConformerTrainer
)
