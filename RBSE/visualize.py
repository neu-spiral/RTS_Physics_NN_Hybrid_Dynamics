# visualize.py
# Visualization for RBSE training results.
#
# Plots:
# 1. State estimates vs true trajectory (with confidence intervals)
# 2. Weight evolution over time
# 3. RMSE bar chart
# 4. Phase portraits (for 2D and 3D systems)
# 5. Epoch comparison (if multi-epoch)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_state_estimates(t, y_true, states_est, state_cov_diag,
                         state_labels, meas_idx, replaced_idx,
                         out_dir, model_name, y_meas=None):
    """
    Plot state estimates vs ground truth with 2-sigma confidence bands.
    """
    n_states = y_true.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(12, 3 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, state_labels)):
        # True trajectory
        ax.plot(t, y_true[:, i], "k-", linewidth=1.2, label="True", alpha=0.8)

        # CKF estimate
        ax.plot(t, states_est[:, i], "b-", linewidth=1.0, label="CKF estimate", alpha=0.9)

        # 2-sigma confidence band
        if state_cov_diag is not None:
            std = np.sqrt(np.maximum(state_cov_diag[:, i], 0))
            ax.fill_between(t, states_est[:, i] - 2*std, states_est[:, i] + 2*std,
                            alpha=0.15, color="blue", label=r"$\pm 2\sigma$")

        # Measurements (if this state is measured)
        if y_meas is not None and i in meas_idx:
            meas_col = meas_idx.index(i)
            # Subsample for readability
            step = max(1, len(t) // 200)
            ax.plot(t[::step], y_meas[::step, meas_col], "r.", markersize=2,
                    alpha=0.4, label="Measurements")

        tag = ""
        if i == replaced_idx:
            tag = " [LEARNED]"
        elif i not in meas_idx:
            tag = " [unmeasured]"
        ax.set_ylabel(label + tag, fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time", fontsize=11)
    fig.suptitle(f"RBSE State Estimation: {model_name.upper()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "state_estimates.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: state_estimates.png")


def plot_estimation_error(t, y_true, states_est, state_cov_diag,
                          state_labels, replaced_idx, out_dir, model_name):
    """
    Plot estimation error with confidence bounds.
    """
    n_states = y_true.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(12, 2.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, state_labels)):
        error = states_est[:, i] - y_true[:, i]
        ax.plot(t, error, "b-", linewidth=0.8, alpha=0.8)

        if state_cov_diag is not None:
            std = np.sqrt(np.maximum(state_cov_diag[:, i], 0))
            ax.fill_between(t, -2*std, 2*std, alpha=0.15, color="red",
                            label=r"$\pm 2\sigma$")

        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")

        tag = " [LEARNED]" if i == replaced_idx else ""
        ax.set_ylabel(f"Error: {label}{tag}", fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time", fontsize=11)
    fig.suptitle(f"RBSE Estimation Error: {model_name.upper()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "estimation_error.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: estimation_error.png")


def plot_weight_evolution(t, weights_est, out_dir, model_name, max_traces=50):
    """
    Plot evolution of NN weights over time.
    Shows a subset of weights to keep the plot readable.
    """
    n_w = weights_est.shape[1]
    n_show = min(n_w, max_traces)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Subplot 1: Individual weight traces
    ax = axes[0]
    indices = np.linspace(0, n_w - 1, n_show, dtype=int)
    for idx in indices:
        ax.plot(t, weights_est[:, idx], linewidth=0.5, alpha=0.6)
    ax.set_ylabel("Weight value", fontsize=11)
    ax.set_title(f"NN Weight Traces ({n_show}/{n_w} shown)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Weight statistics
    ax = axes[1]
    w_mean = np.mean(weights_est, axis=1)
    w_std = np.std(weights_est, axis=1)
    w_norm = np.linalg.norm(weights_est, axis=1)

    ax.plot(t, w_norm, "b-", linewidth=1.2, label="||w||")
    ax.plot(t, w_mean, "g-", linewidth=1.0, alpha=0.8, label="mean(w)")
    ax.fill_between(t, w_mean - w_std, w_mean + w_std, alpha=0.15, color="green",
                    label=r"mean $\pm$ std")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Weight statistics", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"RBSE Weight Evolution: {model_name.upper()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weight_evolution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: weight_evolution.png")


def plot_rmse_bar(rmse_dict, replaced_idx, state_labels, out_dir, model_name):
    """
    Bar chart of per-state RMSE.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(rmse_dict.keys())
    values = [rmse_dict[l] for l in labels]
    colors = ["tab:red" if state_labels.index(l) == replaced_idx else "tab:blue"
              for l in labels]

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(f"RBSE RMSE: {model_name.upper()} (red = learned state)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rmse_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: rmse_bar.png")


def plot_phase_portraits(y_true, states_est, state_labels, out_dir, model_name):
    """
    Phase portraits for pairs of states.
    """
    n_states = y_true.shape[1]
    if n_states < 2:
        return

    # Select pairs to plot
    pairs = []
    if n_states == 2:
        pairs = [(0, 1)]
    elif n_states == 3:
        pairs = [(0, 1), (0, 2), (1, 2)]
    elif n_states == 4:
        pairs = [(0, 1), (0, 2), (2, 3)]
    else:
        # Just a few for high-dimensional systems
        pairs = [(0, 1), (0, 2)]

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4.5))
    if n_pairs == 1:
        axes = [axes]

    for ax, (i, j) in zip(axes, pairs):
        ax.plot(y_true[:, i], y_true[:, j], "k-", linewidth=1.0, alpha=0.6, label="True")
        ax.plot(states_est[:, i], states_est[:, j], "b--", linewidth=0.8,
                alpha=0.8, label="CKF est.")
        # Start markers
        ax.plot(y_true[0, i], y_true[0, j], "ko", markersize=6)
        ax.plot(states_est[0, i], states_est[0, j], "bs", markersize=5)

        ax.set_xlabel(state_labels[i], fontsize=11)
        ax.set_ylabel(state_labels[j], fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"RBSE Phase Portraits: {model_name.upper()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_portraits.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: phase_portraits.png")


def plot_epoch_comparison(t, y_true, all_epoch_states, state_labels,
                          replaced_idx, out_dir, model_name):
    """
    Compare state estimates across epochs for the learned state.
    """
    if all_epoch_states is None or len(all_epoch_states) < 2:
        return

    n_epochs = len(all_epoch_states)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Learned state across epochs
    ax = axes[0]
    ax.plot(t, y_true[:, replaced_idx], "k-", linewidth=1.5, label="True", alpha=0.8)
    for ep, states in enumerate(all_epoch_states):
        alpha = 0.3 + 0.7 * (ep / max(1, n_epochs - 1))
        ax.plot(t, states[:, replaced_idx], linewidth=1.0, alpha=alpha,
                label=f"Epoch {ep+1}")
    ax.set_ylabel(state_labels[replaced_idx], fontsize=11)
    ax.set_title(f"Learned State ({state_labels[replaced_idx]}) Across Epochs", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # RMSE per epoch for learned state
    ax = axes[1]
    rmses = []
    for states in all_epoch_states:
        err = states[:, replaced_idx] - y_true[:, replaced_idx]
        rmses.append(np.sqrt(np.mean(err**2)))
    ax.plot(range(1, n_epochs + 1), rmses, "bo-", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(f"RMSE of {state_labels[replaced_idx]} Per Epoch", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"RBSE Epoch Progress: {model_name.upper()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "epoch_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: epoch_comparison.png")


def plot_rollout(t_rollout, y_true_rollout, y_rollout, state_labels,
                 replaced_idx, meas_idx, out_dir, model_name,
                 rollout_rmse=None):
    """
    Plot standalone rollout of the learned hybrid ODE vs ground truth.

    After CKF training, the final NN weights are extracted and the hybrid
    ODE is integrated forward from the true initial condition. This plot
    shows whether the learned model can reproduce the dynamics without
    any measurement corrections.

    Parameters
    ----------
    t_rollout : (T,) ndarray
    y_true_rollout : (T, n_states) ndarray
    y_rollout : (T, n_states) ndarray
    state_labels : list of str
    replaced_idx : int
    meas_idx : list of int
    out_dir : str
    model_name : str
    rollout_rmse : dict or None
    """
    n_states = y_true_rollout.shape[1]

    # --- Time series plot ---
    fig, axes = plt.subplots(n_states, 1, figsize=(12, 3 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, state_labels)):
        ax.plot(t_rollout, y_true_rollout[:, i], "k-", linewidth=1.2,
                label="True", alpha=0.8)
        color = "tab:red" if i == replaced_idx else "tab:blue"
        ax.plot(t_rollout, y_rollout[:, i], "--", color=color, linewidth=1.0,
                label="Rollout (learned)", alpha=0.9)

        tag = ""
        if i == replaced_idx:
            tag = " [LEARNED]"
        elif i not in meas_idx:
            tag = " [unmeasured]"
        rmse_str = ""
        if rollout_rmse and label in rollout_rmse:
            rmse_str = f"  RMSE={rollout_rmse[label]:.4f}"
        ax.set_ylabel(label + tag, fontsize=11)
        ax.legend(loc="upper right", fontsize=8,
                  title=rmse_str if rmse_str else None)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time", fontsize=11)
    fig.suptitle(f"RBSE Rollout: {model_name.upper()} (standalone integration)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rollout_time_series.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: rollout_time_series.png")

    # --- Rollout phase portraits ---
    if n_states >= 2:
        pairs = []
        if n_states == 2:
            pairs = [(0, 1)]
        elif n_states == 3:
            pairs = [(0, 1), (0, 2), (1, 2)]
        elif n_states == 4:
            pairs = [(0, 1), (0, 2), (2, 3)]
        else:
            pairs = [(0, 1), (0, 2)]

        n_pairs = len(pairs)
        fig, axes_ph = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4.5))
        if n_pairs == 1:
            axes_ph = [axes_ph]

        for ax, (i, j) in zip(axes_ph, pairs):
            ax.plot(y_true_rollout[:, i], y_true_rollout[:, j],
                    "k-", linewidth=1.0, alpha=0.6, label="True")
            ax.plot(y_rollout[:, i], y_rollout[:, j],
                    "r--", linewidth=0.8, alpha=0.8, label="Rollout")
            ax.plot(y_true_rollout[0, i], y_true_rollout[0, j],
                    "ko", markersize=6)
            ax.plot(y_rollout[0, i], y_rollout[0, j],
                    "rs", markersize=5)
            ax.set_xlabel(state_labels[i], fontsize=11)
            ax.set_ylabel(state_labels[j], fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"RBSE Rollout Phase: {model_name.upper()}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "rollout_phase_portraits.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: rollout_phase_portraits.png")

    # --- Rollout error plot ---
    fig, axes_e = plt.subplots(n_states, 1, figsize=(12, 2.5 * n_states), sharex=True)
    if n_states == 1:
        axes_e = [axes_e]

    for i, (ax, label) in enumerate(zip(axes_e, state_labels)):
        error = y_rollout[:, i] - y_true_rollout[:, i]
        color = "tab:red" if i == replaced_idx else "tab:blue"
        ax.plot(t_rollout, error, "-", color=color, linewidth=0.8, alpha=0.8)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        tag = " [LEARNED]" if i == replaced_idx else ""
        ax.set_ylabel(f"Err: {label}{tag}", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes_e[-1].set_xlabel("Time", fontsize=11)
    fig.suptitle(f"RBSE Rollout Error: {model_name.upper()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rollout_error.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: rollout_error.png")


def plot_all(t, y_true, y_meas, states_est, state_cov_diag, weights_est,
             state_labels, meas_idx, replaced_idx, rmse_dict, out_dir,
             model_name, all_epoch_states=None,
             rollout_data=None):
    """
    Generate all plots for RBSE results.
    """
    print(f"\n  Generating plots...")

    plot_state_estimates(t, y_true, states_est, state_cov_diag,
                         state_labels, meas_idx, replaced_idx,
                         out_dir, model_name, y_meas)

    plot_estimation_error(t, y_true, states_est, state_cov_diag,
                          state_labels, replaced_idx, out_dir, model_name)

    plot_weight_evolution(t, weights_est, out_dir, model_name)

    plot_rmse_bar(rmse_dict, replaced_idx, state_labels, out_dir, model_name)

    plot_phase_portraits(y_true, states_est, state_labels, out_dir, model_name)

    if all_epoch_states is not None:
        plot_epoch_comparison(t, y_true, all_epoch_states, state_labels,
                              replaced_idx, out_dir, model_name)

    if rollout_data is not None:
        plot_rollout(
            t_rollout=rollout_data["t"],
            y_true_rollout=rollout_data["y_true"],
            y_rollout=rollout_data["y_rollout"],
            state_labels=state_labels,
            replaced_idx=replaced_idx,
            meas_idx=meas_idx,
            out_dir=out_dir,
            model_name=model_name,
            rollout_rmse=rollout_data.get("rmse"),
        )

    print(f"  All plots saved to: {out_dir}")
