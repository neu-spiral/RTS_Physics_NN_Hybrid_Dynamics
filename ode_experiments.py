"""Command-line entry point for RTS-guided neural-ODE experiments.

Model-specific settings belong to the selected model preset in ``models/``.
This module intentionally exposes only controls that make sense for every
experiment family.
"""

from experiments.ode_training.environment import configure_process_environment

configure_process_environment()

import argparse
import sys

from experiments.ode_training.configuration import VALID_MODELS
from experiments.ode_training.environment import device
from experiments.ode_training.session import train_experiment
from models import make_default_experiment


def _build_parser():
    """Return the small, model-independent experiment interface."""
    parser = argparse.ArgumentParser(
        description='Run an RTS-guided neural ODE experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', choices=sorted(VALID_MODELS), default='ho')
    parser.add_argument('--cpu-threads', type=int, default=None, help='Set CPU library thread count before Torch imports.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--cpu', action='store_true', help='Shortcut for --device cpu.')
    parser.add_argument('--run-id', type=int, default=None, help='Optional identifier for parallel runs.')
    parser.add_argument('--rollouts-from-true', action='store_true', help='Seed diagnostics from true rather than smoothed states.')
    parser.add_argument('--loss-mode', choices=['standard', 'cov_weighted'], default='cov_weighted')
    parser.add_argument('--n-epochs', type=int, default=None)
    parser.add_argument('--t-train', type=float, default=None, help='Training duration in seconds.')
    parser.add_argument('--min-horizon', type=int, default=None, help='Minimum rollout horizon in samples.')
    parser.add_argument('--max-horizon', type=int, default=None, help='Maximum rollout horizon in samples.')
    parser.add_argument('--stride', type=int, default=None, help='Spacing between rollout starts in samples.')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--meas-noise-std', type=float, default=None)
    parser.add_argument('--meas-snr', type=float, default=None)
    parser.add_argument('--pseudo-weight', type=float, default=None)
    parser.add_argument('--full-state-weight', type=float, default=None)
    parser.add_argument('--use-pseudo-latent', dest='use_pseudo_latent', action='store_true', default=None)
    parser.add_argument('--no-pseudo-latent', dest='use_pseudo_latent', action='store_false')
    parser.add_argument('--plot-every', type=int, default=None)
    parser.add_argument('--plot-stride', type=int, default=None)
    parser.add_argument('--max-rollouts', type=int, default=None)
    parser.add_argument('--log-every', type=int, default=None)
    parser.add_argument('--training-log-file', type=str, default=None)
    parser.add_argument('--no-code-backup', dest='no_code_backup', action='store_true', default=False)
    parser.add_argument('--code-backup', dest='no_code_backup', action='store_false')
    parser.add_argument('--rts', dest='use_rts_guidance', action='store_true', default=True)
    parser.add_argument('--no-rts', dest='use_rts_guidance', action='store_false')
    parser.add_argument('--smoother-every', type=int, default=None)
    parser.add_argument('--freeze-smoother', dest='freeze_smoother', action='store_true', default=None)
    parser.add_argument('--no-freeze-smoother', dest='freeze_smoother', action='store_false')
    parser.add_argument('--frozen-smoother-path', type=str, default=None)
    parser.add_argument('--frozen-smoother-epoch', type=int, default=None)
    parser.add_argument('--smoother-fast', dest='smoother_fast', action='store_true', default=None)
    parser.add_argument('--no-smoother-fast', dest='smoother_fast', action='store_false')
    parser.add_argument('--smoother-fast-method', choices=['euler', 'rk2'], default=None)
    parser.add_argument('--smoother-amp', dest='smoother_amp', action='store_true', default=None)
    parser.add_argument('--no-smoother-amp', dest='smoother_amp', action='store_false')
    parser.add_argument('--smoother-inference', dest='smoother_inference', action='store_true', default=None)
    parser.add_argument('--no-smoother-inference', dest='smoother_inference', action='store_false')
    parser.add_argument('--amp', dest='amp', action='store_true', default=None)
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--torch-compile', dest='torch_compile', action='store_true', default=None)
    parser.add_argument('--no-torch-compile', dest='torch_compile', action='store_false')
    parser.add_argument('--timing-sync', dest='timing_sync', action='store_true', default=None)
    parser.add_argument('--no-timing-sync', dest='timing_sync', action='store_false')
    parser.add_argument('--init-checkpoint', type=str, default=None)
    parser.add_argument('--continue-on-nonfinite-loss', dest='continue_on_nonfinite_loss', action='store_true', default=None)
    parser.add_argument('--stop-on-nonfinite-loss', dest='continue_on_nonfinite_loss', action='store_false')
    parser.add_argument('--multi-horizon-steps', type=int, default=None)
    parser.add_argument('--intermediate-weight', type=float, default=None)
    return parser


def _apply_universal_overrides(exp, args):
    """Apply run-wide choices without changing a model's definition."""
    for key, value in vars(args).items():
        setattr(exp, f'{key}_arg', value)
    if args.n_epochs is not None:
        exp.train_cfg.n_epochs = max(0, int(args.n_epochs))
    if args.t_train is not None:
        exp.train_cfg.T_train = float(args.t_train)
    if args.min_horizon is not None:
        exp.train_cfg.min_rollout_horizon = max(1, int(args.min_horizon))
    if args.max_horizon is not None:
        exp.train_cfg.max_rollout_horizon = max(1, int(args.max_horizon))
    exp.train_cfg.max_rollout_horizon = max(exp.train_cfg.min_rollout_horizon, exp.train_cfg.max_rollout_horizon)
    if args.stride is not None:
        exp.train_cfg.stride = max(1, int(args.stride))
    if args.lr is not None:
        exp.nn_cfg.lr = float(args.lr)
    if args.weight_decay is not None:
        exp.nn_cfg.weight_decay = float(args.weight_decay)
    if args.grad_clip is not None:
        exp.nn_cfg.grad_clip = float(args.grad_clip)
    if args.meas_snr is not None:
        if args.meas_snr <= 0.0:
            raise ValueError('--meas-snr must be > 0')
        exp.train_cfg.meas_snr = float(args.meas_snr)
        exp.train_cfg.meas_noise_std = 1.0 / float(args.meas_snr)
    elif args.meas_noise_std is not None:
        exp.train_cfg.meas_noise_std = float(args.meas_noise_std)
        exp.train_cfg.meas_snr = None
    if args.pseudo_weight is not None:
        exp.train_cfg.pseudo_weight = float(args.pseudo_weight)
    if args.full_state_weight is not None:
        exp.train_cfg.full_state_weight = float(args.full_state_weight)
    if args.use_pseudo_latent is not None:
        exp.train_cfg.use_pseudo_latent = bool(args.use_pseudo_latent)
    if args.plot_every is not None:
        exp.plot_cfg.show_every = max(0, int(args.plot_every))
    if args.plot_stride is not None:
        exp.plot_cfg.plot_stride = max(1, int(args.plot_stride))
    if args.max_rollouts is not None:
        exp.plot_cfg.max_rollouts = max(1, int(args.max_rollouts))


def main():
    parser = _build_parser()
    args = parser.parse_args([] if 'spyder_kernels' in sys.modules else None)
    if args.cpu:
        args.device = 'cpu'
    exp = make_default_experiment(args.model)
    _apply_universal_overrides(exp, args)
    print('device:', device)
    train_experiment(
        exp,
        rollouts_from_true=args.rollouts_from_true,
        run_id=args.run_id,
        loss_mode=args.loss_mode,
        use_rts_guidance=args.use_rts_guidance,
    )


if __name__ == '__main__':
    main()
