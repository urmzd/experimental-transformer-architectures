"""
v11a_mixed_ops: Five fixed hand-designed primitive ops composed
sequentially per step. No op selection, no routing — the five ops are
applied in a fixed order, each with its own learned weights.

Hidden state is V-dimensional; input is one-hot; no output projection.

The five ops (originally labelled delta/theta/alpha/beta/gamma, but the
EEG labels are decoration — described here by mechanism):

  1. High-decay EMA    — exponential moving average with decay near 1,
                         provides broad long-range context.
  2. Causal linear-attn — decay-weighted linear attention for long-range
                         content retrieval.
  3. Sigmoid gate      — elementwise learned gate, attenuation.
  4. Dense linear layer — active within-position transform.
  5. Low-decay EMA     — exponential moving average with small decay,
                         provides short-range local context.

Per step = ops 1 through 5 applied in sequence with residual connections.

This is structurally similar to v11b_hard_routing but with the op bank
and routing removed: all five ops always fire, in a fixed order. Untested
per README.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings
from torch import Tensor

from core.base import AgiModel


# ---------------------------------------------------------------------------
# High-decay causal EMA (broad-context smoother)
# ---------------------------------------------------------------------------

class HighDecayEMA(nn.Module):
    """Heavy-decay causal EMA with learned per-channel decay rates.

    Provides slow, long-range smoothing of the register state.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.decay_logits = nn.Parameter(torch.full((dim,), 4.0))  # high init = slow decay
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        decay = torch.sigmoid(self.decay_logits)  # (D,) per-channel decay

        # Parallel scan via cumulative weighted sum
        # For each position t: ema[t] = sum_{s<t} decay^(t-s-1) * x[s]
        # Efficient: use log-space cumsum trick
        log_decay = torch.log(decay.clamp(min=1e-6))  # (D,)
        # Weight each position: w[t] = decay^(T-1-t)
        # fp32 positions: bf16 cannot represent integers > 256 exactly, which
        # would corrupt the decay exponents at the training seq_len of 1024.
        steps = torch.arange(T, device=x.device, dtype=torch.float32)
        # weights[t, d] = exp((T-1-t) * log_decay[d])
        weights = torch.exp(steps.flip(0).unsqueeze(-1) * log_decay.unsqueeze(0))  # (T, D)
        # Weighted x, cumsum, then unweight
        wx = x * weights.unsqueeze(0)  # (B, T, D)
        cum = torch.cumsum(wx, dim=1)
        # Shift right (causal: position t sees only s < t)
        cum = F.pad(cum[:, :-1], (0, 0, 1, 0))
        # Unweight
        result = cum / weights.unsqueeze(0).clamp(min=1e-8)
        return result * self.scale


# ---------------------------------------------------------------------------
# CausalMemory: linear-attention projections shared between two decay heads
# ---------------------------------------------------------------------------

class CausalMemory(nn.Module):
    """Linear attention with causal decay; Q/K/V/O projections are shared.

    The step below calls this module twice per step with two different
    decay logits and scales, giving one long-range and one short-range
    head that reuse the same Q/K/V/O weights.
    """

    def __init__(self, vocab_size: int, state_dim: int):
        super().__init__()
        self.q = nn.Linear(vocab_size, state_dim, bias=False)
        self.k = nn.Linear(vocab_size, state_dim, bias=False)
        self.v = nn.Linear(vocab_size, state_dim, bias=False)
        self.o = nn.Linear(state_dim, vocab_size, bias=False)
        for m in [self.q, self.k, self.v, self.o]:
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: Tensor, decay_logit: Tensor, scale: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype

        q = self.q(x.float()).to(dtype)
        k = self.k(x.float()).to(dtype)
        v = self.v(x.float()).to(dtype)

        scores = torch.bmm(q, k.transpose(1, 2))

        decay = torch.sigmoid(decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        retrieved = torch.bmm(scores, v)
        return self.o(retrieved.float()).to(dtype) * scale


# ---------------------------------------------------------------------------
# SigmoidGate: elementwise learned attenuation
# ---------------------------------------------------------------------------

class SigmoidGate(nn.Module):
    """Learned sigmoid gate over register dimensions.

    Compresses state, computes a sigmoid gate, applies elementwise to x.
    Bias initialized so sigmoid ~= 1 (near-identity at init).
    """

    def __init__(self, vocab_size: int, gate_dim: int):
        super().__init__()
        self.down = nn.Linear(vocab_size, gate_dim, bias=False)
        self.up = nn.Linear(gate_dim, vocab_size)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.constant_(self.up.bias, 2.0)  # sigmoid(2) ≈ 0.88, near-identity

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.up(self.down(x.float())))
        return x * gate.to(x.dtype)


# ---------------------------------------------------------------------------
# DenseTransform: within-position MLP bottleneck
# ---------------------------------------------------------------------------

class DenseTransform(nn.Module):
    """Within-position dense transform.

    Down-project, activate, up-project. Standard MLP bottleneck.
    """

    def __init__(self, vocab_size: int, inner_dim: int):
        super().__init__()
        self.down = nn.Linear(vocab_size, inner_dim, bias=False)
        self.up = nn.Linear(inner_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(inner_dim))
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.down(x.float()) + self.bias)
        return self.up(h).to(x.dtype) * self.scale


# ---------------------------------------------------------------------------
# MixedOpsStep: five primitives composed sequentially
# ---------------------------------------------------------------------------

class MixedOpsStep(nn.Module):
    """One step: five primitive ops applied in a fixed order.

    The long-range and short-range linear-attention heads share the same
    Q/K/V/O projection weights (via `memory`); each step owns only its
    two decay logits and two output scales.

    Op order per step:
      1. High-decay EMA          — broad-context smoother.
      2. Linear attn (long decay)  — long-range retrieval.
      3. Sigmoid gate             — elementwise attenuation.
      4. Dense MLP                — within-position transform.
      5. Linear attn (short decay) — short-range retrieval.
    """

    def __init__(self, vocab_size: int, memory: CausalMemory, inner_dim: int,
                 gate_dim: int):
        super().__init__()
        self.long_ema = HighDecayEMA(vocab_size)
        self.memory = memory  # shared, not owned
        self.long_decay = nn.Parameter(torch.tensor(4.0))  # high = long range
        self.long_scale = nn.Parameter(torch.tensor(0.1))
        self.short_decay = nn.Parameter(torch.tensor(1.0))  # low = short range
        self.short_scale = nn.Parameter(torch.tensor(0.1))
        self.gate = SigmoidGate(vocab_size, gate_dim)
        self.transform = DenseTransform(vocab_size, inner_dim)

    def forward(self, x: Tensor) -> Tensor:
        V = x.size(-1)
        x = x + self.long_ema(F.rms_norm(x, (V,)))
        x = x + self.memory(F.rms_norm(x, (V,)), self.long_decay, self.long_scale)
        x = self.gate(x)
        x = x + self.transform(F.rms_norm(x, (V,)))
        x = x + self.memory(F.rms_norm(x, (V,)), self.short_decay, self.short_scale)
        return x


# ---------------------------------------------------------------------------
# MixedOpsLM
# ---------------------------------------------------------------------------

class MixedOpsLM(AgiModel):
    """Five fixed hand-designed primitive ops composed sequentially per step."""

    version = "v11a_mixed_ops"
    architecture = "Fixed sequential primitives"
    cross_position = "High-decay EMA + causal-decay linattn + low-decay EMA"
    within_position = "Sigmoid gate + dense transform"

    class Settings(BaseSettings):
        vocab_size: int = 1024
        num_steps: int = 8
        state_dim: int = 64
        inner_dim: int = 128
        gate_dim: int = 64
        logit_softcap: float = 30.0

    @classmethod
    def build_kwargs(cls, args) -> dict:
        kw = cls._read_args(args)
        if 'gate_dim' not in kw and hasattr(args, 'state_dim'):
            kw['gate_dim'] = args.state_dim
        return cls._filter_init(kw)

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 state_dim: int = 64, inner_dim: int = 128,
                 gate_dim: int = 64, logit_softcap: float = 30.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        # Shared linear-attention projections (both decay heads share Q/K/V/O across all steps)
        self.memory = CausalMemory(vocab_size, state_dim)

        self.steps = nn.ModuleList([
            MixedOpsStep(vocab_size, self.memory, inner_dim, gate_dim)
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
