"""Final model artifacts, summaries, and training return values."""

from __future__ import annotations

from .environment import *
from .configuration import *

def finalize_run(self):
    session = self
    if self.saw_nonfinite_loss:
        print(f'\n[{self.exp.system_name}] training done with non-finite loss on epoch(s): {self.nonfinite_epochs}.\n')
    else:
        print(f'\n[{self.exp.system_name}] training done.\n')
    import matplotlib.pyplot as plt
    import pandas as pd
    if self.log_file.exists():
        self.df = pd.read_csv(self.log_file)
        self.agg = {'loss': 'mean', 'total_rmse': 'mean', 'horizon': 'first', 'duration_s': 'sum'}
        self.df_grouped = self.df.groupby('epoch').agg(self.agg).reset_index()
        (self.fig, self.axes) = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle(f'{self.title_prefix_rep}: Training Metrics', fontsize=16, fontweight='bold')
        self.axes[0, 0].plot(self.df_grouped['epoch'], self.df_grouped['loss'], 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].set_title('Training Loss Over Time')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].legend()
        self.axes[0, 1].plot(self.df_grouped['epoch'], self.df_grouped['total_rmse'], 'r-', linewidth=2, marker='o', markersize=4, label='Total RMSE')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('RMSE')
        self.axes[0, 1].set_title('State Estimation RMSE Over Time')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].legend()
        self.axes[1, 0].plot(self.df_grouped['epoch'], self.df_grouped['horizon'], 'g-', linewidth=2, marker='s', markersize=4, label='Rollout Horizon')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Horizon Steps')
        self.axes[1, 0].set_title('Training Horizon Schedule')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].legend()
        self.axes[1, 1].plot(self.df_grouped['epoch'], self.df_grouped['duration_s'], 'purple', linewidth=2, marker='^', markersize=4, label='Epoch Duration')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Duration (s)')
        self.axes[1, 1].set_title('Training Time Per Epoch')
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].legend()
        plt.tight_layout()
        self.df_grouped['epoch'] = pd.to_numeric(self.df_grouped['epoch'], errors='coerce')
        self.df_grouped = self.df_grouped.dropna(subset=['epoch'])
        self.last_logged_epoch = int(self.df_grouped['epoch'].max()) if len(self.df_grouped) > 0 else max(0, self.last_completed_epoch)
        self.last_epoch_folder = self.run_root / f'epoch{self.last_logged_epoch:03d}'
        self.last_epoch_folder.mkdir(parents=True, exist_ok=True)
        self.curves_path = self.last_epoch_folder / 'training_curves.png'
        self.fig.canvas.draw()
        plt.savefig(str(self.curves_path), dpi=300, bbox_inches='tight')
        print(f'Training curves saved to {self.curves_path}')
        plt.close(self.fig)
    if self.artifact_system_name == self.exp.system_name:
        self.save_dir = 'saved_models'
    else:
        self.save_dir = os.path.join('saved_models', self.artifact_system_name)
    os.makedirs(self.save_dir, exist_ok=True)
    if self.run_id is not None:
        self.model_path = os.path.join(self.save_dir, f'{self.artifact_system_name}_model_run{self.run_id}.pt')
    else:
        self.model_path = os.path.join(self.save_dir, f'{self.artifact_system_name}_model.pt')
    torch.save(self._checkpoint_payload(self.tc.n_epochs), self.model_path)
    print(f'Model saved to {self.model_path}')
    try:
        self.cfg_path = self.run_root / 'config.txt'
        with open(self.cfg_path, 'w') as self.f:
            self.f.write('=== Experiment Config ===\n')
            self.f.write(f'system: {self.exp.system_name}\n')
            self.f.write(f'title_prefix: {self.exp.title_prefix}\n')
            self.f.write(f'T_train: {self.tc.T_train}\n')
            self.f.write(f'meas_noise_std: {self.tc.meas_noise_std}\n')
            if hasattr(self.tc, 'meas_snr') and self.tc.meas_snr:
                self.f.write(f'meas_snr: {self.tc.meas_snr}\n')
            self.f.write(f'n_epochs: {self.tc.n_epochs}\n')
            self.f.write(f'rollout_horizon: [{self.tc.min_rollout_horizon}, {self.tc.max_rollout_horizon}]\n')
            self.f.write(f'stride: {self.tc.stride}\n')
            self.f.write('\n=== Guidance/Loss ===\n')
            self.f.write(f'loss_mode: {self.loss_mode}\n')
            self.f.write(f'guidance_mode: {self.guidance_label}\n')
            self.f.write(f'use_pseudo_latent: {self.tc.use_pseudo_latent}\n')
            self.f.write(f'pseudo_weight: {self.tc.pseudo_weight}\n')
            self.f.write(f"full_state_weight: {getattr(self.tc, 'full_state_weight', 0.0)}\n")
            self.f.write('\n=== Integrator ===\n')
            self.f.write(f'default_dt: {self.ic.default_dt}\nmax_dt: {self.ic.max_dt}\nmin_dt: {self.ic.min_dt}\nn_substeps: {self.ic.n_substeps}\n')
            self.f.write('\n=== CKF ===\n')
            self.f.write(f'Q_diag: {self.ck.Q_diag}\nR_diag: {self.ck.R_diag}\nP0_diag: {self.ck.P0_diag}\n')
            self.f.write('\n=== NN ===\n')
            self.f.write(f'in_dim: {self.nn_in_dim}\nout_dim: {self.nn_out_dim}\nhidden_dims: {self.nnc.hidden_dims}\nactivation: {self.nnc.activation}\n')
            self.f.write(f'input_scale: {self.nnc.input_scale}\ninput_shift: {self.nnc.input_shift}\n')
            self.f.write(f'lr: {self.nnc.lr}\nweight_decay: {self.nnc.weight_decay}\ngrad_clip: {self.nnc.grad_clip}\n')
            self.f.write('\n=== Data ===\n')
            self.f.write(f'uniform_dt: {self.dc.uniform_dt}  # seconds; 0 means solver-native adaptive grid when keep_adaptive_grid=True\n')
            self.f.write(f"keep_adaptive_grid: {bool(getattr(self.dc, 'keep_adaptive_grid', False))}\n")
            self.f.write(f'max_step_scipy: {self.dc.max_step_scipy}  # max step for solve_ivp; 0=default\n')
            self.f.write(f"scipy_method: {getattr(self.dc, 'scipy_method', '')}\n")
            self.f.write(f'seed: {self.dc.seed}\n')
            self.f.write('\n=== Plot ===\n')
            self.f.write(f'plot_stride: {self.pc.plot_stride}\nshow_every: {self.pc.show_every}\nmax_rollouts: {self.pc.max_rollouts}\n')
            self.f.write('\n=== System Config ===\n')
            self.f.write(str(self.exp.system_cfg))
        print(f'Config summary saved to {self.cfg_path}')
    except Exception as e:
        print(f'[warn] could not save config summary: {self.e}')
    return {'run_root': str(_safe_resolve_path(self.run_root)), 'run_tag': self.ts_tag, 'model_path': str(_safe_resolve_path(self.model_path)), 'config_path': str(_safe_resolve_path(self.cfg_path)) if 'cfg_path' in locals() else None, 'backup_dir': str(_safe_resolve_path(self.backup_dir)) if self.backup_enabled else None, 'loss_mode': self.loss_mode, 'guidance_mode': self.guidance_label, 'use_pseudo_latent': bool(self.tc.use_pseudo_latent), 'pseudo_weight': float(self.tc.pseudo_weight), 'final_loss': self.final_loss, 'final_total_rmse': self.final_total_rmse, 'final_state_rmse': self.final_state_errors, 'final_phase_errors': self.final_phase_errors}
