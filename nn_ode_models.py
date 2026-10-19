# nn_ode_models.py
import torch
import torch.nn as nn
from typing import Optional, Sequence, Union

torch.set_default_dtype(torch.float32)

class StateODEMLP(nn.Module):
    """
    Generic MLP to represent the ENTIRE ODE right-hand side for one or more
    state components.

    Examples
    --------
    - Harmonic oscillator (unknown acceleration):
          dv/dt = NN([x, v])

    - Bladder / retinal:
          dNBa/dt  = NN([state, inputs])
          dX_sub/dt = NN(features)

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output dimension (number of state derivatives this network outputs).
    hidden_dims : int or sequence of int
        Sizes of hidden layers.
    activation : str
        'tanh' or 'relu' (can be extended).
    input_scale : Optional[torch.Tensor]
        Optional (in_dim,) tensor. If given, input is multiplied by this
        before MLP, after optional shift.
    input_shift : Optional[torch.Tensor]
        Optional (in_dim,) tensor. If given, input is first shifted:
            x <- x - input_shift.
        Useful for centering around typical operating points.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Union[int, Sequence[int]] = (32, 32),
        activation: str = "tanh",
        input_scale: Optional[torch.Tensor] = None,
        input_shift: Optional[torch.Tensor] = None,#todo remove
    ):
        # Initialize this object and store the configuration it needs later.
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        # Choose the nonlinearity used by the small derivative model inside the ODE.
        act_name = activation.lower()
        if act_name == "tanh":
            Act = nn.Tanh
        elif act_name == "relu":
            Act = nn.ReLU
        else:
            raise ValueError("Unsupported activation: {}".format(activation))

        # Build the MLP that maps current state features to one or more time derivatives.
        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(Act())
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Keep normalization in the scripted module so f_full can call the NN directly.
        if input_shift is not None:
            self.register_buffer("input_shift", input_shift.view(1, -1))
        else:
            self.input_shift = None

        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.view(1, -1))
        else:
            self.input_scale = None

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        state_features : (..., in_dim)
            Current ODE state features passed to the derivative network.
            For masked systems, non-dependency channels have already been zeroed.
        Returns:
            (..., out_dim)
            One or more learned time derivatives, not next-state values.
        """
        if self.input_shift is not None:
            state_features = state_features - self.input_shift
        if self.input_scale is not None:
            state_features = state_features * self.input_scale
        return self.net(state_features)


def build_scripted_state_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Union[int, Sequence[int]] = (32, 32),
    activation: str = "tanh",
    input_scale: Optional[torch.Tensor] = None,
    input_shift: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.jit.ScriptModule:
    """
    Build the small derivative network used inside hybrid ODE wrappers.

    The returned module is TorchScripted so CKF/RTS propagation and rollout
    losses can call it from scripted integrators without leaving the tensor
    execution path. Optional input_shift/input_scale buffers keep state
    normalization bundled with the saved model.

    Example
    -------
        dv_net = build_scripted_state_mlp(
            in_dim=2,      # [x, v]
            out_dim=1,     # dv/dt
            hidden_dims=(32, 32),
            activation="tanh",
            input_scale=torch.tensor([1.0, 0.5]),
            device=device,
        )
    """
    model = StateODEMLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_scale=input_scale,
        input_shift=input_shift,
    )
    if device is not None:
        model = model.to(device)
    return torch.jit.script(model)
