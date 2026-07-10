from types import SimpleNamespace

import pytest
import torch.nn as nn

from glassbox_lm.core.registry import get_registry, build_model

# Small args for fast instantiation on CPU
_SMALL_ARGS = SimpleNamespace(
    vocab_size=32, num_steps=2,
    n_channels=8, n_fourier_basis=4,
    logit_softcap=30.0, decay_init=3.0, activation="gelu",
    # v1_shared_attn
    num_heads=2, num_kv_heads=2, rope_base=10000.0, qk_gain_init=1.5,
    # v2_conv
    kernel_size=4,
    # v4_weight_shared
    unique_steps=2, invocations_per_step=1, n_heads=2, transform_rank=4,
    # v6_banded_fourier
    band_split="1,1,2", slow_decay_init=4.0, fast_decay_init=2.0,
    # v7_soft_ops / v10_state_cond_op
    n_ops=4,
    # v8_lowrank_vv
    interaction_rank=8,
    # v9_linattn / v11a_mixed_ops / v11b_hard_routing
    state_dim=8, inner_dim=16,
    # v12_vocab_slice / v13_with_embedding
    k_active=16, inner_mul=2, parallel_waves=True, grad_checkpoint=False,
    # v11b_hard_routing
    gumbel_tau=1.0, halt_threshold=0.5, ponder_lambda=0.01,
)


@pytest.mark.parametrize("version", list(get_registry().keys()))
def test_registry_import(version):
    """Every REGISTRY key imports and instantiates successfully."""
    model = build_model(version, _SMALL_ARGS)
    assert isinstance(model, nn.Module)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model version"):
        build_model("nonexistent_v99", _SMALL_ARGS)
