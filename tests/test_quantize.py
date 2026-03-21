import torch

from core.quantize import dequantize_state_dict_int8, quantize_state_dict_int8


def test_roundtrip():
    """Quantize → dequantize produces values close to originals."""
    sd = {
        "large_weight": torch.randn(128, 128),
        "small_bias": torch.randn(16),
    }
    qobj, stats = quantize_state_dict_int8(sd)
    restored = dequantize_state_dict_int8(qobj)

    assert set(restored.keys()) == set(sd.keys())
    # Large tensor goes through int8 quantization — check approximate equality
    torch.testing.assert_close(
        restored["large_weight"], sd["large_weight"],
        atol=0.05, rtol=0.05,
    )
    # Small tensor is passed through (kept as-is or cast to fp16)
    assert restored["small_bias"].shape == sd["small_bias"].shape


def test_stats():
    sd = {"w": torch.randn(512, 256)}
    _, stats = quantize_state_dict_int8(sd)
    assert stats["param_count"] == 512 * 256
    assert stats["num_tensors"] == 1
    assert stats["num_float_tensors"] == 1
