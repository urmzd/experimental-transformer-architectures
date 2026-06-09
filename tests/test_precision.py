import pytest
import torch

from core.quantize import is_control_tensor
from core.registry import get_registry, build_model
from tests.test_models import _SMALL_ARGS


def test_projection_weights_are_not_control_tensors():
    """A substring "weight" pattern once matched every nn.Linear weight and kept
    whole models in fp32. Pin the classification so it cannot regress."""
    for name in (
        "steps.0.attn.q_proj.weight",
        "steps.0.transform.down.weight",
        "hops.0.interaction.U",
        "hops.0.interaction.V",
        "embed.weight",
    ):
        assert not is_control_tensor(name), f"{name} must train in bf16"


def test_control_tensor_suffix_match():
    assert is_control_tensor("hops.0.interaction.diag")
    assert is_control_tensor("steps.0.attn.o_proj.bias")
    assert is_control_tensor("mem_scale")
    # Suffix match, not substring: pattern occurring mid-name does not count.
    assert not is_control_tensor("diag_proj.weight")


@pytest.mark.parametrize("version", list(get_registry().keys()))
def test_train_cast_leaves_no_fp32_weights(version):
    """Apply train.py's exact cast sequence and assert the precision regime:
    non-control ndim>=2 params are bf16, control/small params are fp32."""
    model = build_model(version, _SMALL_ARGS).bfloat16()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (p.ndim < 2 or is_control_tensor(name)) and p.dtype != torch.float32:
                p.data = p.data.float()

    for name, p in model.named_parameters():
        if p.ndim < 2 or is_control_tensor(name):
            assert p.dtype == torch.float32, f"{name} should stay fp32, got {p.dtype}"
        else:
            assert p.dtype == torch.bfloat16, f"{name} should be bf16, got {p.dtype}"
