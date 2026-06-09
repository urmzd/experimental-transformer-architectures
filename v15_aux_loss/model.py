"""
v15_aux_loss: Extends v12_vocab_slice with three training-side additions:
per-step auxiliary cross-entropy loss, top-k sparsification of the
register state after each step, and entropy-conditional write scaling.

Hidden state is V-dimensional; input is one-hot; no output projection.

Per step, in addition to the v12 body:
  1. Aux CE loss — intermediate register state is treated as logits and
     cross-entropy against the target is computed; these losses are
     summed into the total training loss. Gives a direct gradient signal
     at every depth (deep-supervision; related to predictive-coding-style
     per-layer losses).
  2. Top-k sparsification — after each step, only the top-k register
     activations are kept; the rest are zeroed. Forces the model to
     commit to a sparse set of candidates per position.
  3. Entropy-conditional write scaling — per-position write magnitude is
     scaled inversely with the current state entropy, so confident
     positions receive less additional update and uncertain positions
     receive more.

Untested on our benchmark. Each addition is a standard technique; names
("predictive coding", "cortical sparse coding") are rhetorical framings,
not load-bearing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings
from torch import Tensor

from core.base import AgiModel


class CausalDecayMemory(nn.Module):
    """Cross-position mixing via causal decay."""

    def __init__(self, dim: int, decay_init: float = 3.0):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(m.weight, std=0.02)

        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        B, T, k = x.shape
        dtype = x.dtype
        scale = 1.0 / math.sqrt(self.dim)

        q = self.q_proj(x.float()).to(dtype)
        k_ = self.k_proj(x.float()).to(dtype)
        v = self.v_proj(x.float()).to(dtype)

        scores = torch.bmm(q, k_.transpose(1, 2)) * scale

        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        retrieved = torch.bmm(scores, v)
        return self.o_proj(retrieved.float()).to(dtype) * self.out_scale.to(dtype)


class PredictiveRegisterStep(nn.Module):
    """One instruction with sparse state enforcement.

    Returns both the updated state AND a k-dim output for sparse write-back.
    The caller handles residual connections and sparsity enforcement.
    """

    def __init__(self, vocab_size: int, k_active: int, inner_mul: int = 2,
                 activation: str = "gelu", decay_init: float = 3.0,
                 step_idx: int = 0, total_steps: int = 1,
                 sparsity_k: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.k_active = k_active
        self.sparsity_k = sparsity_k

        stride = vocab_size // total_steps if total_steps > 0 else vocab_size
        read_offset = (step_idx * stride) % vocab_size
        write_offset = (read_offset + vocab_size // 2) % vocab_size

        read_indices = torch.tensor(
            [(read_offset + j) % vocab_size for j in range(k_active)],
            dtype=torch.long)
        write_indices = torch.tensor(
            [(write_offset + j) % vocab_size for j in range(k_active)],
            dtype=torch.long)
        write_selector = torch.zeros(k_active, vocab_size)
        write_selector[torch.arange(k_active), write_indices] = 1.0

        self.register_buffer("read_indices", read_indices)
        self.register_buffer("write_selector", write_selector)

        self.memory = CausalDecayMemory(k_active, decay_init)

        inner_dim = k_active * inner_mul
        self.down = nn.Linear(k_active, inner_dim, bias=False)
        self.up = nn.Linear(inner_dim, k_active, bias=False)
        self.mlp_bias = nn.Parameter(torch.zeros(inner_dim))
        self.activation = activation

        nn.init.normal_(self.down.weight, std=0.1)
        nn.init.normal_(self.up.weight, std=0.1)

        self.inv_sqrt_k = 1.0 / math.sqrt(k_active)
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.write_scale = nn.Parameter(torch.tensor(0.1))

    def _mlp(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        h = self.down(x.float()) + self.mlp_bias
        if self.activation == "relu2":
            h = F.relu(h).square()
        elif self.activation == "swish":
            h = F.silu(h)
        else:
            h = F.gelu(h)
        return self.up(h).to(dtype)

    def _enforce_sparsity(self, x: Tensor) -> Tensor:
        """Top-k sparsity with straight-through estimator."""
        if self.sparsity_k >= x.size(-1):
            return x
        topk_vals, topk_idx = x.abs().topk(self.sparsity_k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_idx, 1.0)
        # Straight-through: mask in forward, pass gradients through
        return x * mask + (x - x.detach()) * (1 - mask)

    def forward(self, x: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype
        k = self.k_active

        read_idx = self.read_indices.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        gathered = torch.gather(x, -1, read_idx)

        g_norm = F.rms_norm(gathered, (k,))
        gathered = gathered + self.mem_scale.to(dtype) * self.memory(g_norm)

        g_norm = F.rms_norm(gathered, (k,))
        output = self._mlp(g_norm)

        # Entropy-adaptive write scaling: attenuate writes when state is confident
        with torch.no_grad():
            probs = F.softmax(x.float(), dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(-1, keepdim=True)  # (B, T, 1)
            max_entropy = math.log(V)
            entropy_scale = (entropy / max_entropy).to(dtype)  # 0=confident, 1=uniform

        output = output * (self.write_scale.to(dtype) * self.inv_sqrt_k * entropy_scale)

        delta = output @ self.write_selector.to(dtype)
        x = x + delta

        # Enforce sparsity on the register state
        x = self._enforce_sparsity(x)

        return x


class AuxLossLM(AgiModel):
    """v12 body + per-step auxiliary CE loss + top-k sparsification + entropy-scaled writes."""

    version = "v15_aux_loss"
    architecture = "Aux-loss / sparse-state"
    cross_position = "Decay-weighted linattn in k-slice (sparse state)"
    within_position = "MLP + per-step CE + entropy-scaled writes"

    class Settings(BaseSettings):
        vocab_size: int = 1024
        num_steps: int = 8
        k_active: int = 256
        inner_mul: int = 2
        logit_softcap: float = 30.0
        activation: str = "gelu"
        decay_init: float = 3.0
        sparsity_k: int = 128
        aux_loss_weight: float = 0.1
        aux_loss_decay: float = 0.9

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 k_active: int = 256, inner_mul: int = 2,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 3.0, sparsity_k: int = 128,
                 aux_loss_weight: float = 0.1, aux_loss_decay: float = 0.9):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_decay = aux_loss_decay

        self.steps = nn.ModuleList([
            PredictiveRegisterStep(
                vocab_size, k_active, inner_mul, activation, decay_init,
                step_idx=i, total_steps=num_steps, sparsity_k=sparsity_k)
            for i in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def _compute_logits(self, x: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        flat_targets = target_ids.reshape(-1)
        aux_loss = 0.0

        for i, step in enumerate(self.steps):
            x = step(x)

            # Per-step auxiliary loss (predictive coding)
            if self.aux_loss_weight > 0:
                decay_weight = self.aux_loss_decay ** (self.num_steps - 1 - i)
                step_logits = self._compute_logits(x)
                step_loss = F.cross_entropy(
                    step_logits.float().reshape(-1, V), flat_targets,
                    reduction="mean")
                aux_loss = aux_loss + decay_weight * step_loss

        # Final loss
        final_logits = self._compute_logits(x)
        main_loss = F.cross_entropy(
            final_logits.float().reshape(-1, V), flat_targets, reduction="mean")

        return main_loss + self.aux_loss_weight * aux_loss
