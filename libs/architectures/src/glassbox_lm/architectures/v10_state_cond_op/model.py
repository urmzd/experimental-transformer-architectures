"""
v10_state_cond_op: State-conditioned op dispatch. At each step, a small
MLP reads the current register state and emits the parameters that
determine which registers to read from, which op from a shared bank to
apply, and where to write the result.

Hidden state is V-dimensional; input is one-hot; no output projection.

Per step:
  Observe state -> policy MLP -> emit:
    - read-weights over V (softmax; which input dims to gather)
    - op-weights over N_OPS (softmax; which bank op to apply)
    - write-weights over V (softmax; which output dims to scatter into)
  Gather, apply softly-selected op, scatter back. Cross-position context
  comes from a running memory also read and written through this policy.

Contrast with nearby variants:
  - v7 uses hard-coded per-step instructions with soft Gumbel op selection.
  - v9 learns dense linear-attention projections.
  - v10 factors out the decision-making into an input-dependent MLP that
    parameterizes read/op/write softly at every step, while the op bank
    is shared across steps.

Mechanism is a form of input-conditional mixture-of-experts in vocab
space. Untested per README.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings
from torch import Tensor

from glassbox_lm.core.base import AgiModel


# ---------------------------------------------------------------------------
# Operation bank (shared primitives)
# ---------------------------------------------------------------------------

class OpBank(nn.Module):
    """Bank of primitive operations. State-independent transforms."""

    def __init__(self, n_ops: int, dim: int):
        super().__init__()
        self.n_ops = n_ops
        self.op_transforms = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(n_ops)
        ])
        # Different activation per op
        self.activations = [
            F.gelu,
            F.relu,
            lambda x: F.relu(x).square(),
            F.silu,
            torch.tanh,
            torch.sigmoid,
            lambda x: x,           # identity
            lambda x: -x,          # negate
        ]
        for m in self.op_transforms:
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: Tensor, op_weights: Tensor) -> Tensor:
        """Apply soft-weighted mixture of operations.

        x: (B, T, d) input
        op_weights: (B, T, n_ops) state-dependent selection weights
        """
        result = torch.zeros_like(x)
        for i, (transform, act) in enumerate(zip(self.op_transforms, self.activations)):
            h = act(transform(x.float())).to(x.dtype)
            result = result + op_weights[:, :, i:i+1] * h
        return result


# ---------------------------------------------------------------------------
# Policy network: state → action
# ---------------------------------------------------------------------------

class ActionPolicy(nn.Module):
    """Small MLP that observes register state and outputs action parameters.

    Input: compressed register state (d-dimensional)
    Output: read_weights, op_selection, write_weights

    This is π(action | state) — the core of the RL analogy.
    The policy is LEARNED via backprop, not RL. But the structure
    is RL-inspired: observe state, choose action, execute.
    """

    def __init__(self, state_dim: int, n_ops: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = state_dim * 2
        # Output: read weights (state_dim) + op selection (n_ops) + write weights (state_dim)
        out_dim = state_dim + n_ops + state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.state_dim = state_dim
        self.n_ops = n_ops
        nn.init.normal_(self.net[0].weight, std=0.02)
        nn.init.normal_(self.net[2].weight, std=0.02)

    def forward(self, state: Tensor):
        """
        state: (B, T, state_dim)
        returns: read_w (B,T,d), op_w (B,T,n_ops), write_w (B,T,d)
        """
        out = self.net(state.float()).to(state.dtype)
        d, k = self.state_dim, self.n_ops
        read_logits = out[..., :d]
        op_logits = out[..., d:d+k]
        write_logits = out[..., d+k:]

        read_w = torch.softmax(read_logits, dim=-1)
        op_w = torch.softmax(op_logits, dim=-1)
        write_w = torch.sigmoid(write_logits)  # gate, not distribution

        return read_w, op_w, write_w


# ---------------------------------------------------------------------------
# Policy step: observe → decide → act
# ---------------------------------------------------------------------------

class PolicyStep(nn.Module):
    """One cycle: compress state → policy decides action → execute → expand back.

    Cross-position: causal decay memory (same family as v9_linattn)
    Within-position: state-dependent action via policy network
    """

    def __init__(self, vocab_size: int, state_dim: int, n_ops: int,
                 op_bank: OpBank, decay_init: float = 3.0):
        super().__init__()
        # Compress/expand between vocab and state space
        self.compress = nn.Linear(vocab_size, state_dim, bias=False)
        self.expand = nn.Linear(state_dim, vocab_size, bias=False)
        nn.init.normal_(self.compress.weight, std=0.02)
        nn.init.normal_(self.expand.weight, std=0.02)

        # Policy: observes compressed state, outputs action
        self.policy = ActionPolicy(state_dim, n_ops)

        # Op bank (shared, not owned)
        self.op_bank = op_bank

        # Cross-position: causal decay memory in state space
        self.mem_q = nn.Linear(state_dim, state_dim, bias=False)
        self.mem_k = nn.Linear(state_dim, state_dim, bias=False)
        self.mem_v = nn.Linear(state_dim, state_dim, bias=False)
        for m in [self.mem_q, self.mem_k, self.mem_v]:
            nn.init.normal_(m.weight, std=0.02)
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))

        self.mem_scale = nn.Parameter(torch.tensor(0.1))
        self.act_scale = nn.Parameter(torch.tensor(0.1))

    def _cross_position(self, h: Tensor) -> Tensor:
        """Causal decay memory in state space."""
        B, T, d = h.shape
        dtype = h.dtype

        q = self.mem_q(h.float()).to(dtype)
        k = self.mem_k(h.float()).to(dtype)
        v = self.mem_v(h.float()).to(dtype)

        scores = torch.bmm(q, k.transpose(1, 2))

        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=h.device)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        return torch.bmm(scores, v)

    def forward(self, x: Tensor) -> Tensor:
        V = x.size(-1)
        dtype = x.dtype

        # Compress to state space
        h = self.compress(F.rms_norm(x, (V,)).float()).to(dtype)  # (B, T, d)

        # Cross-position: memory retrieval
        mem_out = self._cross_position(h)
        h = h + self.mem_scale.to(dtype) * mem_out

        # Policy observes state, decides action
        read_w, op_w, write_w = self.policy(h)

        # Execute action
        # READ: state-dependent selection of which dims matter
        selected = h * read_w

        # OP: state-dependent operation selection
        transformed = self.op_bank(selected, op_w)

        # WRITE: state-dependent gating of what to store
        result = transformed * write_w

        # Expand back to vocab space
        return self.expand(result.float()).to(dtype) * self.act_scale.to(dtype)


# ---------------------------------------------------------------------------
# StateCondOpLM
# ---------------------------------------------------------------------------

class StateCondOpLM(AgiModel):
    """State-conditioned soft dispatch over read / op-bank / write at each step."""

    version = "v10_state_cond_op"
    architecture = "State-conditioned op dispatch"
    cross_position = "Linear attention in compressed state"
    within_position = "State-conditioned soft read + op + write"

    class Settings(BaseSettings):
        vocab_size: int = 1024
        num_steps: int = 8
        state_dim: int = 64
        n_ops: int = 8
        logit_softcap: float = 30.0
        decay_init: float = 3.0

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 state_dim: int = 64, n_ops: int = 8,
                 logit_softcap: float = 30.0, decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        # Shared op bank (same primitives available to all steps)
        self.op_bank = OpBank(n_ops, state_dim)

        # Each step has its own policy and compress/expand
        self.steps = nn.ModuleList([
            PolicyStep(vocab_size, state_dim, n_ops, self.op_bank, decay_init)
            for _ in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for step in self.steps:
            x = x + step(x)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
