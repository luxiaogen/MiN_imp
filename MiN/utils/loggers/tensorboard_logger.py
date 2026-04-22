import logging
import os

from .base import BaseExperimentLogger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


class TensorBoardLogger(BaseExperimentLogger):
    def __init__(self, log_dir, flush_secs=10):
        self.log_dir = log_dir
        self.writer = None

        if SummaryWriter is None:
            logging.warning(
                "TensorBoard logger requested but tensorboard is not installed. "
                "Falling back to a no-op logger."
            )
            return

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

    def log_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag, values, step):
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def log_text(self, tag, text, step=0):
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
