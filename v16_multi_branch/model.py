"""
v16_multi_branch: Three structural additions on top of the base recurrent
vocab-space machine: parallel independent columns whose outputs are
combined by learned weights; a branched MLP inside each step with input-
dependent gating across branches; and learned cross-column inhibitory
connections.

Hidden state per column is V-dimensional; input is one-hot; no output
projection. Columns are ensembled; the final prediction is a learned
linear combination of column states.

Per step, per column:
  - Branched MLP: B parallel sub-MLPs take the same input; their outputs
    are combined by an input-dependent gate (softmax or sigmoid over B),
    which lets each branch specialize on a sub-region of input space.
  - Column output is a standard residual update.

Across columns (per step):
  - A learned (C x C) inhibition matrix subtracts a scaled combination of
    sibling columns' states from each column's state, encouraging
    columns to carry distinct information.

Notes: originally framed with Hawkins / Thousand-Brains / dendritic-
computation language. The actual mechanisms are ensembling, gated MLP
branches, and learned cross-stream interference — standard building
blocks, not biological commitments. Untested per README.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.base import AgiModel, CommonSettings


def make_fourier_basis(dim: int, n_basis: int) -> Tensor:
    pos = torch.arange(dim, dtype=torch.float32) / dim
    basis = torch.zeros(dim, 2 * n_basis)
    for k in range(n_basis):
        freq = k + 1
        basis[:, 2 * k] = torch.cos(2 * math.pi * freq * pos)
        basis[:, 2 * k + 1] = torch.sin(2 * math.pi * freq * pos)
    return basis


class CausalDecayMemory(nn.Module):
    """Cross-position mixing via causal decay in sparse subspace."""

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


class DendriticMLP(nn.Module):
    """MLP with multiple dendritic branches and input-dependent gating.

    Instead of one hidden layer, N branches compute independently,
    then are combined via a learned gate conditioned on the input.
    This provides nonlinear interaction capacity that a single wide
    layer cannot achieve.
    """

    def __init__(self, dim: int, inner_mul: int = 2, n_branches: int = 4,
                 activation: str = "gelu"):
        super().__init__()
        self.n_branches = n_branches
        inner_dim = dim * inner_mul

        # Each branch has its own down/up projection
        self.branch_down = nn.ModuleList([
            nn.Linear(dim, inner_dim // n_branches, bias=False)
            for _ in range(n_branches)
        ])
        self.branch_up = nn.ModuleList([
            nn.Linear(inner_dim // n_branches, dim, bias=False)
            for _ in range(n_branches)
        ])
        self.bias = nn.Parameter(torch.zeros(inner_dim // n_branches))

        # Gate: input-dependent weighting of branches
        self.gate_proj = nn.Linear(dim, n_branches, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.ones_(self.gate_proj.bias)  # Start with equal weighting

        self.activation = activation
        for m in self.branch_down + self.branch_up:
            nn.init.normal_(m.weight, std=0.1)

    def _act(self, x: Tensor) -> Tensor:
        if self.activation == "relu2":
            return F.relu(x).square()
        elif self.activation == "swish":
            return F.silu(x)
        return F.gelu(x)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        # Compute branch outputs
        branch_outs = []
        for down, up in zip(self.branch_down, self.branch_up):
            h = self._act(down(x.float()) + self.bias)
            branch_outs.append(up(h).to(dtype))

        # Input-dependent gating
        gate_logits = self.gate_proj(x.float())  # (B, T, n_branches)
        gates = F.softmax(gate_logits, dim=-1).to(dtype)

        # Weighted combination
        result = torch.zeros_like(branch_outs[0])
        for i, bout in enumerate(branch_outs):
            result = result + gates[..., i:i+1] * bout

        return result


class ColumnStep(nn.Module):
    """One step within a cortical column."""

    def __init__(self, vocab_size: int, k_active: int, inner_mul: int = 2,
                 n_branches: int = 4, activation: str = "gelu",
                 decay_init: float = 3.0, step_idx: int = 0,
                 total_steps: int = 1):
        super().__init__()
        self.k_active = k_active

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
        self.mlp = DendriticMLP(k_active, inner_mul, n_branches, activation)

        self.inv_sqrt_k = 1.0 / math.sqrt(k_active)
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.write_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype
        k = self.k_active

        read_idx = self.read_indices.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        gathered = torch.gather(x, -1, read_idx)

        g_norm = F.rms_norm(gathered, (k,))
        gathered = gathered + self.mem_scale.to(dtype) * self.memory(g_norm)

        g_norm = F.rms_norm(gathered, (k,))
        output = self.mlp(g_norm)

        output = output * (self.write_scale.to(dtype) * self.inv_sqrt_k)
        return output @ self.write_selector.to(dtype)


class CorticalColumn(nn.Module):
    """One independent processing column with its own steps."""

    def __init__(self, vocab_size: int, num_steps: int, k_active: int,
                 inner_mul: int, n_branches: int, activation: str,
                 decay_init: float, column_idx: int, total_columns: int):
        super().__init__()
        self.steps = nn.ModuleList([
            ColumnStep(
                vocab_size, k_active, inner_mul, n_branches, activation,
                decay_init,
                step_idx=i * total_columns + column_idx,
                total_steps=num_steps * total_columns)
            for i in range(num_steps)
        ])
        # Column confidence: learned scalar indicating how much this column
        # should contribute to the final vote
        self.confidence = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        for step in self.steps:
            x = x + step(x)
        return x


class MultiBranchLM(AgiModel):
    """Multi-column ensemble + input-gated branched MLP + cross-column inhibition."""

    version = "v16_multi_branch"
    architecture = "Multi-column / branched MLP"
    cross_position = "Per-column decay-weighted linattn"
    within_position = "Input-gated branched MLP"

    class Settings(CommonSettings):
        num_columns: int = 4
        steps_per_column: int = 3
        k_active: int = 128
        inner_mul: int = 2
        n_branches: int = 4

    def __init__(self, vocab_size: int = 1024, num_columns: int = 4,
                 steps_per_column: int = 3, k_active: int = 128,
                 inner_mul: int = 2, n_branches: int = 4,
                 n_fourier_basis: int = 16, n_channels: int = 128,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_columns = num_columns
        self.logit_softcap = logit_softcap

        self.columns = nn.ModuleList([
            CorticalColumn(
                vocab_size, steps_per_column, k_active, inner_mul,
                n_branches, activation, decay_init,
                column_idx=c, total_columns=num_columns)
            for c in range(num_columns)
        ])

        # Lateral inhibition: columns compete
        self.inhibition = nn.Parameter(
            torch.zeros(num_columns, num_columns))

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x0 = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x0 = F.rms_norm(x0, (V,))

        # Each column processes independently
        column_outputs = [col(x0) for col in self.columns]

        # Confidence-weighted voting with lateral inhibition
        raw_confidences = torch.stack(
            [col.confidence for col in self.columns])  # (C,)

        # Lateral inhibition: each column's confidence is reduced by others
        inhibited = raw_confidences - (
            self.inhibition @ raw_confidences.detach())
        weights = F.softmax(inhibited, dim=0)  # (C,)

        # Weighted vote
        x = torch.zeros_like(column_outputs[0])
        for w, out in zip(weights, column_outputs):
            x = x + w * out

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
