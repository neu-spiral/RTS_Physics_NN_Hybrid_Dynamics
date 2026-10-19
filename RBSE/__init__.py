# RBSE: CKF-based joint state and parameter estimation baseline
# Implements the approach from:
#   "Cubature Kalman Filter Based Training of Hybrid Differential Equation
#    Recurrent Neural Network Physiological Dynamic Models"
#   (Demirkaya et al., 2021)
#
# Key idea: Augment the state vector with NN weights, so the CKF
# simultaneously estimates both the ODE states and the NN parameters.
# No backprop, no rollouts, no RTS smoother needed.

# Importing RBSE exposes its public functions but never starts a run.  Resolve
# them lazily so ``python RBSE/train.py`` does not import that file twice.
__all__ = ('main', 'run_rbse')


def __getattr__(name):
    if name in __all__:
        from RBSE.train import main, run_rbse
        return {'main': main, 'run_rbse': run_rbse}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
