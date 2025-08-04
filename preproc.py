import logging
from typing import Type

from omegaconf import DictConfig, OmegaConf

from common.config import PreprocArgs
from common.log import setup_log
from common.path import get_conf_file_path
from data.processor.builder import EEGDatasetBuilder
from data.processor.wrapper import DATASET_SELECTOR


logger = logging.getLogger('preproc')


def prepare_dataset(
        conf: PreprocArgs,
        builder_cls: Type[EEGDatasetBuilder],
        dataset_name: str,
        config_name: str
):
    try:
        logger.info(f"Preparing dataset {dataset_name} {config_name}...")
        builder = builder_cls(config_name)
        if conf.clean_middle_cache:
            builder.clean_disk_cache()
        builder.preproc(n_proc=conf.num_preproc_mid_workers)
        builder.download_and_prepare(num_proc=conf.num_preproc_arrow_writers)
        dataset = builder.as_dataset()
        logger.info(f"Dataset {dataset_name} {config_name} is prepared.")
        logger.info(f"{dataset}")
    except Exception as e:
        logger.error(f"Preparation of dataset {dataset_name} {config_name} exit with error: {e}.")


def preproc(conf: PreprocArgs):
    dataset_names = conf.pretrain_datasets
    dataset_configs = ['pretrain' for _ in dataset_names]
    dataset_names.extend(conf.finetune_datasets.keys())
    dataset_configs.extend(conf.finetune_datasets.values())

    for dataset, config in zip(dataset_names, dataset_configs):
        if dataset not in DATASET_SELECTOR.keys():
            raise ValueError(f"Dataset {dataset} is not supported.")

        builder_cls = DATASET_SELECTOR[dataset]
        if config not in builder_cls.builder_configs.keys():
            raise ValueError(f"Config {config} is not supported for dataset {dataset}.")

        prepare_dataset(conf, builder_cls, dataset, config)


if __name__ == '__main__':
    # cli args
    # conf_file = [abs path | rel path | file name]
    cli_args: DictConfig = OmegaConf.from_cli()
    logger.info(cli_args)
    if 'conf_file' in cli_args.keys():
        logger.info(cli_args.conf_file)
        file_cfg = OmegaConf.load(get_conf_file_path(cli_args.conf_file))
        cli_args.pop("conf_file")
    else:
        file_cfg = OmegaConf.create({})

    code_cfg = OmegaConf.create(PreprocArgs().model_dump())
    
    setup_log()
    cfg = OmegaConf.merge(code_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.info(cfg)
    conf = PreprocArgs.model_validate(cfg)

    preproc(conf)
