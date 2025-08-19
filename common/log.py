import sys
import time
import logging
import os.path
from typing import Optional
from logging import Filter, Formatter, StreamHandler


PRECISION_DICT = {
    "lr": "6e",
    "header_lr": "6e",
    "encoder_lr": "6e",
    "gram": "2f",
    "accuracy": "3f",
    "f1": "3f",
    "pr": "3f",
    "recall": "3f",
    "cohen_kappa": "3f",
    "auroc": "3f",
    "auc_pr": "3f",
}


class DistributedTimeFilter(Filter):
    def __init__(self, start_time: Optional[float]=None):
        super().__init__()
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time
        self.is_slurm = get_is_slurm_job()
        self.rank = get_global_rank()

    def filter(self, record):
        delta = int(time.time() - self.start_time)
        hours, remain = divmod(delta, 3600)
        minutes, seconds = divmod(remain, 60)

        record.time_delta = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        record.rank = f"{self.rank}:"
        return True


def format_console_log_dict(log_data: dict, prefix: str = 'train'):
    prefix = f"{prefix}/"
    log_data = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in log_data.items()}
    formatted_log = ", ".join([
        f"{key}: {value:.{PRECISION_DICT.get(key, '5e')}}" if isinstance(value, float)
        else f"{key}: {value}"
        for key, value in log_data.items()
    ])
    formatted_log = f"{prefix[:-1]} {formatted_log}"
    return formatted_log


def sync_deepspeed_log_handler(default_logger: logging.Logger, deepspeed_logger: logging.Logger):
    for handler in deepspeed_logger.handlers[:]:
        deepspeed_logger.removeHandler(handler)
        handler.close()

    for handler in default_logger.handlers:
        if isinstance(handler, StreamHandler):
            deepspeed_logger.addHandler(handler)


def setup_log(
        file_path: Optional[str]=None, *,
        start_time: Optional[float]=None,
        name: Optional[str]=None,
        level: Optional[str]=None
):
    datefmt = "%y-%m-%d %H:%M:%S"
    fmt = "%(rank)s%(levelname)-7s %(asctime)s.%(msecs)03d +%(time_delta)s - %(filename)s:%(lineno)d - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if level is None else level)

    stdout_handler = StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG if level is None else level)
    stderr_handler = StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    handlers = [stdout_handler, stderr_handler]

    formatter = Formatter(fmt, datefmt=datefmt)
    time_filter = DistributedTimeFilter(start_time)

    if file_path is not None and get_is_master():
        path = os.path.dirname(file_path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(file_path, "a")
        file_handler.setLevel(logging.NOTSET)
        handlers.append(file_handler)

    logger.handlers.clear()
    for handler in handlers:
        handler.addFilter(time_filter)
        handler.setFormatter(formatter)
    logger.handlers.extend(handlers)

    return logger


if __name__ == "__main__":
    setup_log()
    logger = logging.getLogger()

    logger.info("Training started")
    time.sleep(2.0)
