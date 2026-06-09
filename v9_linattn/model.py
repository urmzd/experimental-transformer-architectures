"""
v9_linattn: Linear attention with exponential causal decay, using dense
(full-rank) V -> r projections for Q/K/V/O. Within-position mixing is a
dense MLP with activation.

Hidden state is V-dimensional; input is one-hot; no output projection.

This is the same family as v3_fourier_linattn, with one key difference:
the Q/K/V/O projections are unconstrained learned matrices instead of
Fourier-basis-parameterized matrices. That removes the structural
bottleneck that stalls v3. Computationally equivalent to Linear
Transformer / RWKV-style linear attention with decay.

Mechanism:
  output_t = sum_{s < t} decay^(t-s-1) * (q_t . k_s) * v_s
           = causal decay-weighted matmul, fully parallel.
  Written equivalently as an outer-product accumulator S_t that is
  updated once per position; the docstring previously framed this as a
  "Q-table," but it is simply the linear-attention state matrix.

Per step has its own projection weights. Multiple steps = stacked linear
attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings
from torch import Tensor

from core.base import AgiModel


# ---------------------------------------------------------------------------
# Linear attention with causal decay and dense projections
# ---------------------------------------------------------------------------

class LinearAttnHead(nn.Module):
    """Linear attention with exponential causal decay.

    Projects vocab-space to a small state-space via dense learned matrices.
    Computes decay-weighted (q . k) v, fully parallel via decay-masked
    matmul. Equivalent to maintaining an outer-product state that updates
    once per position and is queried by q_t.
    """

    def __init__(self, vocab_size: int, state_dim: int, decay_init: float = 3.0):
        super().__init__()
        self.state_dim = state_dim
        # Dense projections: vocab → state (full-rank, not Fourier)
        self.q_proj = nn.Linear(vocab_size, state_dim, bias=False)
        self.k_proj = nn.Linear(vocab_size, state_dim, bias=False)
        self.v_proj = nn.Linear(vocab_size, state_dim, bias=False)
        self.o_proj = nn.Linear(state_dim, vocab_size, bias=False)

        # Initialize small
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(m.weight, std=0.02)

        # Decay: how quickly old associations fade
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, V)
        returns: (B, T, V) — retrieved associations mapped back to vocab
        """
        B, T, V = x.shape
        dtype = x.dtype

        queries = self.q_proj(x.float()).to(dtype)   # (B, T, r)
        keys = self.k_proj(x.float()).to(dtype)      # (B, T, r)
        values = self.v_proj(x.float()).to(dtype)     # (B, T, r)

        # Causal decay-weighted similarity (parallel)
        scores = torch.bmm(queries, keys.transpose(1, 2))  # (B, T, T)

        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        # Retrieve context
        retrieved = torch.bmm(scores, values)  # (B, T, r)

        # Project back to vocab space
        return self.o_proj(retrieved.float()).to(dtype) * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Register transform: simple within-position operation
# ---------------------------------------------------------------------------

class RegisterTransform(nn.Module):
    """Within-position transform. Dense down-project, activate, up-project.

    No Fourier. Just a small MLP bottleneck in vocab space.
    Down: V → r (compress). Activate: gelu/relu. Up: r → V (expand).
    """

    def __init__(self, vocab_size: int, inner_dim: int, activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        self.down = nn.Linear(vocab_size, inner_dim, bias=False)
        self.up = nn.Linear(inner_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(inner_dim))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        h = self.down(x.float()) + self.bias
        if self.activation == "relu":
            h = F.relu(h)
        elif self.activation == "relu2":
            h = F.relu(h).square()
        elif self.activation == "swish":
            h = F.silu(h)
        else:
            h = F.gelu(h)
        return self.up(h).to(dtype) * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# One recurrent step
# ---------------------------------------------------------------------------

class LinAttnStep(nn.Module):
    """One step: linear-attention head + within-position MLP."""

    def __init__(self, vocab_size: int, state_dim: int, inner_dim: int,
                 activation: str = "gelu", decay_init: float = 3.0):
        super().__init__()
        self.attn = LinearAttnHead(vocab_size, state_dim, decay_init)
        self.transform = RegisterTransform(vocab_size, inner_dim, activation)
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.t_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        V = x.size(-1)
        x = x + self.attn_scale.to(x.dtype) * self.attn(F.rms_norm(x, (V,)))
        x = x + self.t_scale.to(x.dtype) * self.transform(F.rms_norm(x, (V,)))
        return x


# ---------------------------------------------------------------------------
# LinAttnLM
# ---------------------------------------------------------------------------

class LinAttnLM(AgiModel):
    """Linear attention with causal decay, dense projections; MLP within-position."""

    version = "v9_linattn"
    architecture = "Linear attention (dense)"
    cross_position = "Decay-weighted linear attention"
    within_position = "MLP bottleneck"

    class Settings(BaseSettings):
        vocab_size: int = 1024
        num_steps: int = 8
        state_dim: int = 64
        inner_dim: int = 128
        logit_softcap: float = 30.0
        activation: str = "gelu"
        decay_init: float = 3.0

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 state_dim: int = 64, inner_dim: int = 128,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        self.steps = nn.ModuleList([
            LinAttnStep(vocab_size, state_dim, inner_dim, activation, decay_init)
            for _ in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for step in self.steps:
            x = step(x)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
