from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from core.registry import get_registry, build_model

_SMALL_ARGS = SimpleNamespace(
    vocab_size=32, num_steps=2,
    n_channels=8, n_fourier_basis=4,
    logit_softcap=30.0, decay_init=3.0, activation="gelu",
    num_heads=2, num_kv_heads=2, rope_base=10000.0, qk_gain_init=1.5,
    kernel_size=4,
    unique_steps=2, invocations_per_step=1, n_heads=2, transform_rank=4,
    band_split="1,1,2", slow_decay_init=4.0, fast_decay_init=2.0,
    n_ops=4, interaction_rank=8,
    state_dim=8, inner_dim=16,
    k_active=16, inner_mul=2, parallel_waves=True, grad_checkpoint=False,
    gumbel_tau=1.0, halt_threshold=0.5, ponder_lambda=0.01,
)

BATCH, SEQ = 2, 4


@pytest.mark.parametrize("version", list(get_registry().keys()))
def test_forward_returns_scalar_loss(version):
    """Each model forward(input_ids, target_ids) returns a scalar loss."""
    model = build_model(version, _SMALL_ARGS).float()
    input_ids = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))
    target_ids = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))

    loss = model(input_ids, target_ids)

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


@pytest.mark.parametrize("version", list(get_registry().keys()))
def test_backward(version):
    """Each model supports backward pass."""
    model = build_model(version, _SMALL_ARGS).float()
    input_ids = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))
    target_ids = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))

    loss = model(input_ids, target_ids)
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "At least one parameter should have a gradient"


def _forward_logits(model, input_ids):
    """Capture the per-position (B, T, V) logits a forward pass produces.

    Models return only a scalar loss, so the logits are intercepted at the
    F.cross_entropy call every variant ends with. This sees the true output
    after all functional post-processing, including variants whose final state
    never passes through a hookable module (v1/v4/v6/v7 weight-shared loops).
    """
    vocab = _SMALL_ARGS.vocab_size
    real_ce = F.cross_entropy
    captured = []

    def spy(logits, *args, **kwargs):
        if isinstance(logits, torch.Tensor) and logits.numel() == input_ids.numel() * vocab:
            captured.append(logits.detach().reshape(*input_ids.shape, vocab).clone())
        return real_ce(logits, *args, **kwargs)

    F.cross_entropy = spy
    try:
        with torch.no_grad():
            model(input_ids, torch.zeros_like(input_ids))
    finally:
        F.cross_entropy = real_ce
    return captured


@pytest.mark.parametrize("position", [1, SEQ // 2, SEQ - 1])
@pytest.mark.parametrize("version", list(get_registry().keys()))
def test_causal_masking_no_future_leakage(version, position):
    """Changing the input at one position must not move logits anywhere before it.

    A future-token leak in any variant would look like an architecture win on
    next-token metrics, so the prefix is required to be bit-identical, not
    approximately equal. Seeds are reset before each forward so stochastic
    models (v11b Gumbel) draw identical noise for both inputs.
    """
    torch.manual_seed(0)
    model = build_model(version, _SMALL_ARGS).float().eval()
    x1 = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))
    x2 = x1.clone()
    x2[:, position] = (x2[:, position] + 1) % _SMALL_ARGS.vocab_size

    torch.manual_seed(1)
    logits1 = _forward_logits(model, x1)
    torch.manual_seed(1)
    logits2 = _forward_logits(model, x2)

    assert logits1, "no (B, T, V) logits reached F.cross_entropy"
    assert len(logits1) == len(logits2)
    for l1, l2 in zip(logits1, logits2):
        assert torch.equal(l1[:, :position], l2[:, :position]), \
            f"input at position {position} leaked into earlier logits"
    assert any(not torch.equal(l1[:, position:], l2[:, position:])
               for l1, l2 in zip(logits1, logits2)), \
        "perturbation never reached the logits; the test has no power"


def test_v15_sparsity_straight_through():
    """_enforce_sparsity masks to top-k in forward but passes full gradients.

    A plain top-k mask zeroes gradients at every non-top-k position, silently
    killing learning through the sparsity bottleneck — the straight-through
    estimator must keep the backward pass an identity.
    """
    from v15_aux_loss.model import PredictiveRegisterStep

    step = PredictiveRegisterStep(vocab_size=32, k_active=16, sparsity_k=4)
    x = torch.randn(2, 3, 32, requires_grad=True)

    out = step._enforce_sparsity(x)

    assert int((out != 0).sum(-1).max()) <= 4, "forward must keep only top-k"
    out.sum().backward()
    assert torch.all(x.grad == 1.0), \
        "gradient must pass through zeroed positions (straight-through)"
