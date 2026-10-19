"""Generated-data preparation and diagnostics for benchmark experiments."""

from __future__ import annotations

from .environment import *
from .configuration import *


def _adjust_ckf_for_snr(exp):
    """Update measurement covariance when an SNR override is configured."""
    tc = exp.train_cfg
    if not getattr(tc, "meas_snr", None):
        return
    snr = tc.meas_snr
    state_scales = getattr(exp.system_cfg, "state_scales", None)
    meas_idx = list(getattr(exp, "meas_idx", [])) or list(range(len(exp.ckf_cfg.R_diag)))
    r_old = list(exp.ckf_cfg.R_diag)
    if state_scales is not None:
        if len(r_old) == len(meas_idx):
            r_new = r_old[:]
            for j, idx in enumerate(meas_idx):
                sigma = 1.0 / (state_scales[idx] * snr)
                r_new[j] = sigma * sigma
        elif len(r_old) == len(state_scales):
            r_new = r_old[:]
            for idx in meas_idx:
                sigma = 1.0 / (state_scales[idx] * snr)
                r_new[idx] = sigma * sigma
        else:
            print(
                f"[warn] Cannot align R_diag with meas_idx for SNR override "
                f"(len R={len(r_old)}, len meas_idx={len(meas_idx)})"
            )
            return
    else:
        default_std = type(tc)().meas_noise_std
        new_std = 1.0 / snr
        ratio = (new_std / default_std) ** 2 if default_std > 0 else 1.0
        r_new = [value * ratio for value in r_old]
    exp.ckf_cfg = exp.ckf_cfg.__class__(
        Q_diag=exp.ckf_cfg.Q_diag,
        R_diag=tuple(r_new),
        P0_diag=exp.ckf_cfg.P0_diag,
    )
    print(f"[SNR] Adjusted R_diag for meas_snr={snr}: {tuple(f'{r:.2e}' for r in r_new)}")


def _as_float_tensor(value, *, device):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _save_step_size_diagnostic(t: torch.Tensor, out_dir, *, title: str = "Time Grid"):
    """Save a PNG/NPZ diagnostic for solver output times and step sizes."""
    out_dir = _safe_resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_np = t.detach().cpu().numpy()
    dt_np = np.diff(t_np) if t_np.size > 1 else np.asarray([], dtype=np.float64)
    if dt_np.size:
        dt_min = float(np.min(dt_np))
        dt_max = float(np.max(dt_np))
        dt_mean = float(np.mean(dt_np))
        dt_median = float(np.median(dt_np))
        dt_std = float(np.std(dt_np))
        uniform_tol = max(1e-12, abs(dt_mean) * 0.0005)
        is_uniform = bool(dt_max - dt_min <= uniform_tol)
    else:
        dt_min = dt_max = dt_mean = dt_median = dt_std = float("nan")
        uniform_tol = float("nan")
        is_uniform = True

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    fig.suptitle(f"{title}: Solver Step Sizes", fontweight="bold")
    axes[0].plot(np.arange(t_np.size), t_np, marker=".", linewidth=0.9)
    axes[0].set(title="Time Grid", xlabel="sample index", ylabel="t")
    axes[0].grid(True, alpha=0.3)
    if dt_np.size:
        axes[1].plot(np.arange(dt_np.size), dt_np, marker=".", linewidth=0.9)
        axes[1].set(title="Step Sizes", xlabel="step index", ylabel="dt")
        axes[1].grid(True, alpha=0.3)
        positive_dt = dt_np[dt_np > 0.0]
        if positive_dt.size:
            axes[2].plot(np.arange(dt_np.size), dt_np, marker=".", linewidth=0.9)
            axes[2].set_yscale("log")
            axes[2].set(title="Step Sizes (log scale)", xlabel="step index", ylabel="dt")
            axes[2].grid(True, which="both", alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "No positive dt values for log scale", ha="center", va="center")
            axes[2].set_axis_off()
        axes[3].hist(dt_np, bins=min(50, max(5, int(np.sqrt(dt_np.size)))), color="steelblue", alpha=0.85)
        axes[3].set(title="Step Size Distribution", xlabel="dt", ylabel="count")
        axes[3].grid(True, alpha=0.3)
        if positive_dt.size and float(np.max(positive_dt)) > float(np.min(positive_dt)):
            axes[3].set_xscale("log")
    else:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, "No step sizes: fewer than two time points", ha="center", va="center")
            ax.set_axis_off()
    summary = (
        f"N={t_np.size}\nsteps={dt_np.size}\nuniform={is_uniform}\n"
        f"dt_min={dt_min:.8g}\ndt_max={dt_max:.8g}\ndt_mean={dt_mean:.8g}\n"
        f"dt_median={dt_median:.8g}\ndt_std={dt_std:.8g}\nuniform_tol={uniform_tol:.8g}"
    )
    axes[0].text(
        0.02, 0.98, summary, transform=axes[0].transAxes, va="top", ha="left",
        family="monospace", fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    png_path = out_dir / "step_sizes.png"
    npz_path = out_dir / "step_sizes.npz"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    np.savez(npz_path, t=t_np, dt=dt_np)
    print(
        f"[Data] step-size diagnostic saved: {png_path} "
        f"(N={t_np.size}, dt_min={dt_min:.6g}, dt_max={dt_max:.6g}, uniform={is_uniform})"
    )
    return png_path, npz_path


def _save_generated_data_diagnostic(
    t: torch.Tensor,
    true_states: torch.Tensor,
    meas: torch.Tensor,
    meas_idx,
    state_labels,
    out_dir,
    *,
    title: str = "Generated Data",
    drive_seq: torch.Tensor = None,
    filename_stem: str = "generated_data",
):
    """Save canonical generated-data NPZ and PDF at the run root."""
    out_dir = _safe_resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_np = t.detach().cpu().numpy()
    true_np = true_states.detach().cpu().numpy()
    meas_np = meas.detach().cpu().numpy()
    if true_np.ndim == 3 and true_np.shape[1] == 1:
        true_np = true_np[:, 0, :]
    if meas_np.ndim == 1:
        meas_np = meas_np[:, None]
    elif meas_np.ndim == 3 and meas_np.shape[1] == 1:
        meas_np = meas_np[:, 0, :]
    meas_idx_list = list(meas_idx)
    drive_np = drive_seq.detach().cpu().numpy() if drive_seq is not None else None
    npz_path = out_dir / f"{filename_stem}.npz"
    np.savez(
        npz_path,
        t=t_np,
        true_states=true_np,
        measurements=meas_np,
        meas_idx=np.asarray(meas_idx_list, dtype=np.int64),
        state_labels=np.asarray(list(state_labels), dtype=object),
        drive_seq=drive_np if drive_np is not None else np.asarray([], dtype=np.float32),
    )

    import matplotlib.pyplot as plt

    n_states = true_np.shape[1] if true_np.ndim == 2 else 1
    fig, axes = plt.subplots(
        n_states, 1, figsize=(12, max(4.0, min(18.0, 1.8 * n_states))),
        sharex=True, constrained_layout=True,
    )
    axes = [axes] if n_states == 1 else axes
    for idx, ax in enumerate(axes):
        label = state_labels[idx] if idx < len(state_labels) else f"x{idx}"
        y_true = true_np[:, idx] if true_np.ndim == 2 else true_np
        ax.plot(t_np, y_true, color="black", linewidth=1.0, label=f"{label} true")
        if idx in meas_idx_list:
            meas_col = meas_idx_list.index(idx)
            if meas_col < meas_np.shape[1]:
                ax.plot(t_np, meas_np[:, meas_col], ".", color="tab:blue", markersize=2.5, alpha=0.7, label=f"{label} meas")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    if drive_np is not None and drive_np.size:
        ax_drive = axes[-1].twinx()
        ax_drive.plot(t_np, drive_np.reshape(drive_np.shape[0], -1)[:, 0], color="tab:orange", linewidth=1.0, alpha=0.75)
        ax_drive.set_ylabel("drive")
    axes[-1].set_xlabel("time")
    fig.suptitle(title, fontweight="bold")
    pdf_path = out_dir / f"{filename_stem}.pdf"
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Data] generated data saved: {npz_path}")
    print(f"[Data] generated data PDF saved: {pdf_path}")
    return npz_path, pdf_path
