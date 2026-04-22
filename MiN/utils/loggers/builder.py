import os

from .null_logger import NullLogger
from .tensorboard_logger import TensorBoardLogger


def build_experiment_logger(args):
    backend = args.get("experiment_logger", "tensorboard")

    if backend == "none":
        return NullLogger()

    if backend == "tensorboard":
        base_log_dir = args.get("log_dir", "logs")
        subdir = args.get("tensorboard_subdir", "tensorboard")
        log_dir = args.get("tensorboard_log_dir", os.path.join(base_log_dir, subdir))
        flush_secs = args.get("tensorboard_flush_secs", 10)
        return TensorBoardLogger(log_dir=log_dir, flush_secs=flush_secs)

    raise ValueError(f"Unsupported experiment logger backend: {backend}")
