import pytest
import torch

from apps.cli.observe import capture_register_states, step_modules
from core.registry import build_model, get_registry
from tests.test_models import _SMALL_ARGS, BATCH, SEQ


def _passive_states(model, input_ids):
    """Record per-step states with hooks that return None (provably inert)."""
    rec = []

    def hook(_mod, _inp, out):
        o = out[0] if isinstance(out, tuple) else out
        rec.append(o.detach().float().clone())

    handles = [m.register_forward_hook(hook) for _, m in step_modules(model)]
    with torch.no_grad():
        model(input_ids, torch.zeros_like(input_ids))
    for h in handles:
        h.remove()
    return rec


@pytest.mark.parametrize("version", list(get_registry().keys()))
def test_capture_matches_unhooked_forward(version):
    """capture_register_states must observe the forward pass, never alter it.

    Guards against the pre-2026-06-09 bug where the hook returned a value
    unconditionally, replacing tuple module outputs with their first tensor.
    Seeds are reset before each forward so stochastic models (v11b Gumbel)
    draw identical noise.
    """
    torch.manual_seed(0)
    model = build_model(version, _SMALL_ARGS).float().eval()
    input_ids = torch.randint(0, _SMALL_ARGS.vocab_size, (BATCH, SEQ))
    target = torch.zeros_like(input_ids)

    torch.manual_seed(1)
    with torch.no_grad():
        loss_before = model(input_ids, target)

    torch.manual_seed(1)
    _, captured, _ = capture_register_states(model, input_ids)

    torch.manual_seed(1)
    with torch.no_grad():
        loss_after = model(input_ids, target)
    assert torch.equal(loss_before, loss_after), "capture left the model changed"

    torch.manual_seed(1)
    reference = _passive_states(model, input_ids)

    assert len(captured) == len(reference)
    for step, (got, want) in enumerate(zip(captured, reference)):
        assert torch.equal(got, want), f"captured state diverges at step {step}"
