"""Stateful orchestration for an RTS-guided neural-ODE training run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .lifecycle import initial_rollout, prepare_data, prepare_run
from .model_setup import build_network, configure_training, initialize_smoother
from .outputs import finalize_run
from .training_loop import (
    optimize_epoch,
    plot_epoch,
    report_epoch,
    run_epoch_smoother,
    save_epoch_artifacts,
)


@dataclass
class TrainingSession:
    exp: Any
    rollouts_from_true: bool = False
    run_id: str | None = None
    loss_mode: str = "standard"
    use_rts_guidance: bool = True

    def run(self):
        prepare_run(self)
        prepare_data(self)
        build_network(self)
        configure_training(self)
        initialize_smoother(self)
        initial_rollout(self)
        for self.epoch in range(1, self.tc.n_epochs + 1):
            run_epoch_smoother(self)
            optimize_epoch(self)
            report_epoch(self)
            plot_epoch(self)
            if save_epoch_artifacts(self):
                break
        return finalize_run(self)


def train_experiment(exp, rollouts_from_true=False, run_id=None, loss_mode="standard", use_rts_guidance=True):
    return TrainingSession(exp, rollouts_from_true, run_id, loss_mode, use_rts_guidance).run()
