import logging
from typing import Optional

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field


logger = logging.getLogger()


class VisArgs(BaseModel):
    ckpt_path: str = ''
    output_dir: str = ''
    tag: list[str] = Field(default_factory=lambda: [])
    seed: int = 42

    split: str = 'test'
    model_type: str = 'former'
    datasets: dict[str, str] = Field(default_factory=lambda: {})

    def dump_to_yaml(self, path: Optional[str ] =None, sort_keys: bool = False):
        conf = self.model_dump()
        conf_yaml = yaml.dump(
            conf,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=sort_keys
        )

        logger.info('Config is as follows in this run:')
        logger.info(conf_yaml)

        if path is not None:
            with open(path, 'w') as f:
                f.write(conf_yaml)


class IntegratedGradientsVisArgs(VisArgs):
    n_steps: int = 50
    baseline_type: str = 'random'  # 'zero', 'random', 'gaussian'
    ig_target: str = 'channel'  # channel or temporal
    
    # NoiseTunnel
    noise_tunnel_type: str = 'smoothgrad'  # 'smoothgrad', 'smoothgrad_sq', 'vargrad'
    noise_tunnel_samples: int = 25
    noise_tunnel_stdevs: float = 0.2

    num_batch: int = 5
    generate_class_average: bool = True
    generate_per_sample: bool = True


class GradCamVisArgs(VisArgs):
    grad_cam_target: str = 'channel' # channel or temporal
    num_batch: int = 5
    label_option: str = 'pred' # pred or truth
    generate_class_average: bool = True
    generate_per_sample: bool = True


class TsneVisArgs(VisArgs):
    num_batch: int = 500
    perplexity: int = 30
    small_perplexity: int = 10
    use_pca: bool = False
    pca_dims: int = 50
    max_iter: int = 1000


def load_vis_conf_dict(config_path, vis_type: str) -> VisArgs:
    file_cfg = OmegaConf.load(config_path)
    if vis_type == 't_sne':
        config_class = TsneVisArgs
    elif vis_type == 'grad_cam':
        config_class = GradCamVisArgs
    elif vis_type == 'integrated_gradients':
        config_class = IntegratedGradientsVisArgs

    else:
        raise ValueError(f'Unknown vis_type: {vis_type}')

    code_cfg = OmegaConf.create(config_class().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg)
    cfg_dict = OmegaConf.to_container(merged_config, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg_dict)

    return cfg
