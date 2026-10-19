"""Continue neural-ODE training from frozen CKF/RTS smoother outputs.

The implementation lives in :mod:`experiments.smoother_continuation`; this
command file owns the continuation-specific command-line interface.
"""

import argparse

from experiments.smoother_continuation.support import DEFAULT_CHECKPOINT
from experiments.smoother_continuation.session import train_from_smoother

def main():
    parser = argparse.ArgumentParser(description='Train NN from frozen smoother checkpoint')
    parser.add_argument('--checkpoint', '-c', default=DEFAULT_CHECKPOINT, help=f'Path to a checkpoint .pt file or epoch directory. Default: {DEFAULT_CHECKPOINT}')
    parser.add_argument('--smoother-checkpoint', type=str, default=None, help='Optional checkpoint to use only for frozen smoother tensors. Useful when the NN checkpoint and the correct ms_s/Ps_s source differ.')
    parser.add_argument('--n-epochs', '-n', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: from checkpoint)')
    parser.add_argument('--min-horizon', type=int, default=None, help='Min rollout horizon')
    parser.add_argument('--max-horizon', type=int, default=None, help='Max rollout horizon')
    parser.add_argument('--stride', type=int, default=None, help='Rollout stride')
    parser.add_argument('--crop-start', type=float, default=None, help='Optional first time to keep from the checkpoint')
    parser.add_argument('--crop-end', type=float, default=None, help='Optional last time to keep from the checkpoint')
    parser.add_argument('--multi-horizon-steps', type=int, default=None, help='Number of rollout horizons per epoch')
    parser.add_argument('--rollout-repeats', type=int, default=None, help="How many times to repeat each epoch's rollout-horizon set")
    parser.add_argument('--rollout-max-dt', type=float, default=None, help='Override the integrator max_dt used for training rollouts and saved checkpoints.')
    parser.add_argument('--tbptt-steps', type=int, default=None, help='Truncated BPTT depth in rollout samples (default: 5).')
    parser.add_argument('--long-horizon-boost', type=int, default=None, help='Extra optimization passes at the longest rollout horizon')
    parser.add_argument('--artifact-root', type=str, default='.', help='Base directory where new results/ and saved_models/ trees will be created (default: repo root).')
    parser.add_argument('--run-suffix', type=str, default=None, help='Optional run suffix used in <timestamp>_<suffix>_fromsmoother, e.g. nom.')
    parser.add_argument('--long-rollout-steps', type=int, default=None, help='Fixed horizon in steps for the long saved rollout visualization.')
    parser.add_argument('--plot-rollout-steps', type=int, default=None, help='Fixed horizon in steps for the short saved rollout visualization.')
    parser.add_argument('--plot-max-rollouts', type=int, default=None, help='Maximum rollout start points to draw in saved rollout visualizations.')
    parser.add_argument('--plot-stride', type=int, default=None, help='Stride between candidate rollout start points for saved rollout visualizations.')
    parser.add_argument('--plot-max-dt', type=float, default=None, help='Maximum internal dt used by dense rollout plotting and BDF max_step.')
    parser.add_argument('--skip-full-ode-plot', action='store_true', help='Skip the expensive full-window model-vs-physics plot; rollout plots are still saved.')
    parser.add_argument('--no-plots', action='store_true', help='Skip diagnostic plots during training')
    parser.add_argument('--loss-mode', choices=['standard', 'cov_weighted'], default=None)
    parser.add_argument('--full-state-weight', type=float, default=None, help='Weight for matching every frozen smoother state.')
    parser.add_argument('--pseudo-weight', type=float, default=None, help='Weight for the learned latent-state smoother target.')
    parser.add_argument('--plot-stiff-solver', choices=['rk4', 'bdf', 'radau'], default='rk4', help='Rollout visualization backend: native RK4 or a stiff reference solver')
    parser.add_argument('--no-resume-optimizer', action='store_true', help='Do not restore optimizer state')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--device', type=str, default=None, help='Device (auto/cpu/cuda)')
    args = parser.parse_args()
    train_from_smoother(args.checkpoint, args)


if __name__ == '__main__':
    main()
