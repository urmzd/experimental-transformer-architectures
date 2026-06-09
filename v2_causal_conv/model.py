"""
v2_conv: Depthwise causal 1D convolution across positions + Fourier-
parameterized within-position channel mix, applied to a vocab-dimensional
state.

Hidden state is V-dimensional; input is one-hot; no output projection.

Per step (each step has its own weights — no sharing):
  x = x + scale_i * DepthwiseCausalConv_i(rms_norm(x))
  x = x + scale_i * FourierOp_i(rms_norm(x))

DepthwiseCausalConv: one filter per V dimension, kernel_size taps back in
position, left-padded to preserve causality. Captures "how active was
dimension j over the last k positions" without cross-dimension mixing.
Cross-dimension mixing happens in the Fourier op (see caveat below).

FourierOp: two V x C projections parameterized as linear combinations of a
fixed sin/cos basis over vocab indices. The basis imposes a smoothness
prior over vocab-id ordering — which is arbitrary, since vocab ids come
from BPE. Functionally a low-rank V -> C -> V mixer with a non-generic
parameterization.

Result per README: 353K params, val_loss 5.39, strong baseline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.base import AgiModel
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Depthwise causal convolution (replaces attention)
# ---------------------------------------------------------------------------

class DepthwiseCausalConv1D(nn.Module):
    """Causal depthwise conv over sequence positions.

    Each vocabulary dimension is convolved independently along the sequence.
    This captures "how active was word j in the last k positions?"
    Cross-word mixing is handled by the FourierRegisterOp.

    For dim=1024, kernel_size=16: only 16K params (vs 3M for attention).
    """

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(dim, 1, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) → (B, D, T) for conv1d
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))  # causal: pad left only
        x = F.conv1d(x, self.weight.to(x.dtype), self.bias.to(x.dtype),
                     groups=x.size(1))
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# Fourier register operations
# ---------------------------------------------------------------------------

def make_fourier_basis(dim: int, n_basis: int) -> Tensor:
    """Fourier basis functions over vocabulary indices."""
    pos = torch.arange(dim, dtype=torch.float32) / dim
    basis = torch.zeros(dim, 2 * n_basis)
    for k in range(n_basis):
        freq = k + 1
        basis[:, 2 * k] = torch.cos(2 * math.pi * freq * pos)
        basis[:, 2 * k + 1] = torch.sin(2 * math.pi * freq * pos)
    return basis


class FourierRegisterOp(nn.Module):
    """One LGP instruction operating on the vocabulary register bank.

    Read:  gather word activations via Fourier-weighted patterns
    Mix:   channel transform + nonlinearity
    Write: scatter result back to vocabulary registers
    """

    def __init__(self, n_basis: int, n_channels: int, activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        s = 0.02
        self.read_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.write_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.channel_mix = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        read_w = torch.softmax(basis @ self.read_coeffs.T, dim=0)
        values = x @ read_w.to(x.dtype)
        values = values @ self.channel_mix.to(x.dtype) + self.bias.to(x.dtype)
        if self.activation == "relu2":
            values = F.relu(values).square()
        elif self.activation == "swish":
            values = F.silu(values)
        else:
            values = F.gelu(values)
        write_w = (basis @ self.write_coeffs.T).to(x.dtype)
        return values @ write_w.T * self.out_scale.to(x.dtype)


# ---------------------------------------------------------------------------
# Register step (one LGP instruction)
# ---------------------------------------------------------------------------

class RegisterStep(nn.Module):
    """One complete instruction: cross-position conv + within-position transform."""

    def __init__(self, dim: int, kernel_size: int, n_basis: int,
                 n_channels: int, activation: str = "gelu"):
        super().__init__()
        self.conv = DepthwiseCausalConv1D(dim, kernel_size)
        self.register_op = FourierRegisterOp(n_basis, n_channels, activation)
        self.conv_scale = nn.Parameter(torch.ones(dim))
        self.op_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        D = x.size(-1)
        x = x + self.conv_scale.to(x.dtype) * self.conv(F.rms_norm(x, (D,)))
        x = x + self.op_scale.to(x.dtype) * self.register_op(
            F.rms_norm(x, (D,)), basis)
        return x


# ---------------------------------------------------------------------------
# CausalConvLM
# ---------------------------------------------------------------------------

class CausalConvLM(AgiModel):
    """Depthwise causal 1D conv + per-step Fourier channel mix in vocab space."""

    version = "v2_conv"
    architecture = "Causal convolution"
    cross_position = "Depthwise causal convolution"
    within_position = "Fourier-parameterized channel mix"

    class Settings(BaseSettings):
        vocab_size: int = 1024
        num_steps: int = 48
        kernel_size: int = 16
        n_fourier_basis: int = 16
        n_channels: int = 64
        logit_softcap: float = 30.0
        activation: str = "gelu"

    def __init__(self, vocab_size: int = 1024, num_steps: int = 48,
                 kernel_size: int = 16, n_fourier_basis: int = 16,
                 n_channels: int = 64, logit_softcap: float = 30.0,
                 activation: str = "gelu"):
        super().__init__()
        dim = vocab_size
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        self.steps = nn.ModuleList([
            RegisterStep(dim, kernel_size, n_fourier_basis, n_channels, activation)
            for _ in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fourier_basis",
                             make_fourier_basis(dim, n_fourier_basis))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size

        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for step in self.steps:
            x = step(x, self.fourier_basis)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
