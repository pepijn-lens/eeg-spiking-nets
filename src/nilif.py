"""
NiLIF — Non-iterative Leaky Integrate-and-Fire neuron (Zhang et al., arXiv:2312.05643).

Standard LIF couples timesteps recurrently: u[t] = λ(u[t-1] - Vth·o[t-1]) + I[t],
causing vanishing gradients for T > 5. NiLIF unrolls the recurrence analytically:

    u[t] = Σ_{i<=t} I[i]·λ^(t-i)  -  Vth · Σ_{i<t} o[i]·λ^(t-i-1)
           (--------Ein----------)         (-----------Eout----------)

In matrix form: Ein = I @ L_in.T, Eout = o_hat @ L_out.T, where L_in and L_out are
lower-triangular Toeplitz matrices parameterised by the learnable decay λ = exp(-1/τ).
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class _SpikeFunction(torch.autograd.Function):
    """Heaviside forward, piecewise-linear surrogate backward."""

    @staticmethod
    def forward(ctx, u: Tensor, vth: float) -> Tensor:
        ctx.save_for_backward(u)
        ctx.vth = vth
        return (u >= vth).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (u,) = ctx.saved_tensors
        # Rectangular window: gradient = 1 if vth <= u < vth+1, else 0
        mask = ((u >= ctx.vth) & (u < ctx.vth + 1.0)).float()
        return grad_output * mask, None  # None for vth (not a Tensor)


def spike(u: Tensor, vth: float = 0.5) -> Tensor:
    """Spike function with surrogate gradient for backprop."""
    return _SpikeFunction.apply(u, vth)


class NiLIF(nn.Module):
    """
    Non-iterative LIF neuron operating along the last (T) dimension.

    Input:  (B, C, S, T)
    Output: (B, C, S, T) — binary spike tensor in {0.0, 1.0}

    τ is a learnable scalar (stored as log_tau for positivity).
    L_in and L_out are rebuilt every forward pass to stay in the autograd graph.
    """

    def __init__(self, T: int, vth: float = 0.5, init_tau: float = 5.0):
        super().__init__()
        self.T = T
        self.vth = vth
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))

        # Pre-built index difference matrix — device-safe buffer, parameter-independent
        idx = torch.arange(T, dtype=torch.float32)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # diff[t,i] = t-i, shape (T, T)
        self.register_buffer("diff", diff)

    def _build_matrices(self) -> tuple[Tensor, Tensor]:
        tau = self.log_tau.exp()
        lam = torch.exp(-1.0 / tau)
        # L_in[t,i] = λ^(t-i) for t>=i, 0 otherwise
        L_in = lam.pow(self.diff.clamp(min=0)).tril()  # (T, T)
        # L_out: strictly lower triangular — row-shift L_in down by 1
        L_out = torch.zeros_like(L_in)
        L_out[1:] = L_in[:-1]
        return L_in, L_out

    def forward(self, x: Tensor) -> Tensor:
        B, C, S, T = x.shape
        assert T == self.T, f"NiLIF(T={self.T}) received T={T}"

        L_in, L_out = self._build_matrices()  # (T, T) each

        I = x.reshape(B * C * S, T)            # (N, T)

        # Step 1: un-reset membrane estimate
        Ein = I @ L_in.t()                     # (N, T)

        # Step 2: approximate spike train — straight-through estimator (STE).
        # Forward: hard Heaviside (same approximation as the paper).
        # Backward: gradient passes straight through to Ein without a window constraint.
        # Using a second surrogate here (as in the original formulation) compounds with
        # the outer surrogate in step 5 and leaves only ~5% of neurons with non-zero
        # gradient, causing training to stall. STE removes this bottleneck.
        o_hat_hard = (Ein >= self.vth).float()
        o_hat = Ein + (o_hat_hard - Ein).detach()  # STE

        # Step 3: reset contribution from approximate spikes
        Eout = o_hat @ L_out.t()               # (N, T)

        # Step 4: corrected membrane
        U = Ein - self.vth * Eout              # (N, T)

        # Step 5: final spikes with surrogate gradient (single window constraint)
        O = spike(U, self.vth)                 # (N, T)

        return O.reshape(B, C, S, T)
