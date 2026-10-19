"""Stateful orchestration for frozen-smoother continuation training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .loading import load_checkpoint
from .model_setup import build_model, configure_training, prepare_epochs
from .training_loop import finalize, log_epoch, optimize_epoch, plot_epoch, save_epoch


@dataclass
class ContinuationSession:
    checkpoint_path: str
    args: Any

    def run(self):
        load_checkpoint(self)
        build_model(self)
        configure_training(self)
        prepare_epochs(self)
        for self.epoch in range(getattr(self, 'epoch_offset', 0) + 1, self.n_epochs + 1):
            optimize_epoch(self)
            log_epoch(self)
            plot_epoch(self)
            save_epoch(self)
        return finalize(self)


def train_from_smoother(checkpoint_path: str, args):
    return ContinuationSession(checkpoint_path, args).run()
