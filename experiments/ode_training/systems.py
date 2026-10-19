"""Build the physics model and data bundle for each supported system."""

from __future__ import annotations

from .environment import *
from .configuration import *
from .data import *


def _choose_system(exp):
    """Return physics, data, labels, priors, measurements, and drive metadata."""
    ic, ck = exp.integrator_cfg, exp.ckf_cfg
    _apply_ckf_overrides(
        ck,
        q_diag=getattr(exp, "ckf_q_diag_arg", None),
        r_diag=getattr(exp, "ckf_r_diag_arg", None),
        p0_diag=getattr(exp, "ckf_p0_diag_arg", None),
    )
    name = exp.system_name
    if name == "ho":
        t, meas, s0_true, s1_true = make_data_ho(exp, device)
        physics: BasePhysics2D = HOPhysics(exp.system_cfg, ic, ck).to(device)
        return (
            physics, (t, meas, s0_true, s1_true), ["x", "v"], "HO",
            [exp.system_cfg.x0_prior, exp.system_cfg.v0_prior], [0], 2, None,
        )
    if name == "pendulum":
        t, meas, s0_true, s1_true = make_data_pendulum(exp, device)
        physics: BasePhysics2D = PendulumPhysics(exp.system_cfg, ic, ck).to(device)
        return (
            physics, (t, meas, s0_true, s1_true), ["theta", "omega"], "Pendulum",
            [exp.system_cfg.theta0_prior, exp.system_cfg.omega0_prior], [0], 2, None,
        )
    if name == "hh":
        t, meas, v_true, m_true, h_true, n_true = make_data_hh(exp, device)
        physics = HodgkinHuxleyPhysics(exp.system_cfg, ic, ck).to(device)
        true_states = torch.stack([v_true, m_true, h_true, n_true], dim=-1)
        prior = [
            exp.system_cfg.V0_prior, exp.system_cfg.m0_prior,
            exp.system_cfg.h0_prior, exp.system_cfg.n0_prior,
        ]
        return physics, (t, meas, v_true, m_true, true_states), ["V", "m", "h", "n"], "HH", prior, [0, 2, 3], 4, None
    if name == "brusselator":
        t, meas, x_true, y_true, z_true = make_data_brusselator(exp, device)
        physics = BrusselatorPhysics(exp.system_cfg, ic, ck).to(device)
        true_states = torch.stack([x_true, y_true, z_true], dim=-1)
        prior = [exp.system_cfg.x0_prior, exp.system_cfg.y0_prior, exp.system_cfg.z0_prior]
        return physics, (t, meas, x_true, y_true, true_states), ["x", "y", "z"], "Brusselator", prior, [1, 2], 3, None
    if name == "yeast":
        data = make_data_yeast(exp, device)
        t, meas = data[:2]
        state_columns = data[2:9]
        physics = YeastPhysics(exp.system_cfg, ic, ck).to(device)
        true_states = torch.stack(list(state_columns), dim=-1)
        labels = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
        return physics, (t, meas, state_columns[0], state_columns[1], true_states), labels, "Yeast", list(exp.system_cfg.x0_prior), [0, 1, 2, 4, 5, 6], 7, None
    if name == "retina":
        t, measurements, true_states, drive_seq = make_data_retina(exp, device)
        physics = RetinaPhysics(exp.system_cfg, ic, ck).to(device)
        labels = ["P1", "P2", "P4", "P5"]
        prior = [
            exp.system_cfg.p1_prior, exp.system_cfg.p2_prior,
            exp.system_cfg.p4_prior, exp.system_cfg.p5_prior,
        ]
        meas_idx = list(getattr(exp, "meas_idx", (0, 1, 3)))
        title = getattr(exp, "title_prefix", "Retina")
        return physics, (t, measurements, true_states[:, 0], true_states[:, 1], true_states), labels, title, prior, meas_idx, 5, drive_seq
    raise ValueError(f"Unknown system_name '{name}'")


__all__ = [name for name in globals() if not name.startswith("__")]
