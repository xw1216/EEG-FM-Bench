import os
import random
from datetime import datetime
from functools import lru_cache

from common.config import BaseLoggingArgs
from common.path import RUN_ROOT


@lru_cache()
def get_is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()


@lru_cache()
def get_global_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache()
def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache()
def get_is_master() -> bool:
    return get_global_rank() == 0


@lru_cache()
def get_master_port(job_id: int, port: int, is_port_random: bool = False) -> int:
    if get_is_torch_run():
        return int(os.environ["MASTER_PORT"])
    elif not is_port_random:
        return port
    else:
        min_master_port, max_master_port = (20000, 60000)
        rng = random.Random(job_id)
        return rng.randint(min_master_port, max_master_port)


@lru_cache()
def get_master_addr() -> str:
    if get_is_torch_run():
        return os.environ["MASTER_ADDR"]
    elif get_is_slurm_job():
        # hostnames = subprocess.check_output(
        #     ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        # )
        # return hostnames.split()[0].decode("utf-8")
        node_list = os.environ['SLURM_NODELIST']
        if os.path.exists('/ssdfs'):
            return node_list
        else:
            print(node_list)
            addr = node_list[12:].replace('-', '.')
            return addr
    else:
        return "127.0.0.1"

@lru_cache()
def get_available_cpu() -> int:
    slurm_cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        return int(slurm_cpus_per_task)

    return os.cpu_count()

@lru_cache()
def get_specific_dirname() -> str:
    # make sure invoke only once
    if get_is_slurm_job():
        return f'slurm_{os.environ["SLURM_JOB_ID"]}'
    elif get_is_torch_run():
        return f"torchrun_{datetime.now().strftime('%y%m%d%H%M%S')}"
    else:
        return f"local_{datetime.now().strftime('%y%m%d%H%M%S')}"


def get_train_io_path(args: BaseLoggingArgs) -> tuple[str, str]:
    if not get_is_master():
        return '', ''

    name = get_specific_dirname()
    run_dir = args.run_dir if args.run_dir else RUN_ROOT

    log_path = os.path.join(run_dir, 'log', 'train', name)
    ckpt_path = os.path.join(run_dir, 'ckpt', name)

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    return log_path, ckpt_path
