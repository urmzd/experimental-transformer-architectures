from __future__ import annotations

import torch

CONTROL_TENSOR_NAME_PATTERNS = (
    "mem_scale", "op_scale", "read_coeffs", "write_coeffs",
    "channel_mix", "bias", "out_scale", "logit_scale", "decay_logit",
    "coeffs",
    # v4 additions
    "diag", "mix_down", "mix_up", "decay_logits", "_override",
    # gauss additions
    "freq_to_ch", "ch_to_freq", "weight",
    # lgp additions
    "op_logits", "op_weights", "op_biases",
    # graph additions
    "q_scale", "k_scale", "diag", "prop_scale", "interact_scale",
    # tpg additions
    "halt_proj", "act_selector", "scale_gate", "t_scale",
    # sparse register additions (v12)
    "mem_scale", "write_scale", "mlp_bias",
)

INT8_CLIP_Q = 99.99984 / 100.0
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536


def quantize_state_dict_int8(sd):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
                kept = t.float().contiguous()
            elif t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(torch.float16).contiguous()
            else:
                kept = t
            passthrough[name] = kept
            stats["int8_payload_bytes"] += kept.numel() * kept.element_size()
            continue
        stats["num_float_tensors"] += 1
        t32 = t.float()
        if t32.ndim == 2:
            ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            clipped = torch.clamp(t32, -ca[:, None], ca[:, None])
            s = (ca / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / s[:, None]), -127, 127).to(torch.int8)
            qmeta[name] = {"scheme": "per_row", "axis": 0}
            scales[name] = s.to(torch.float16).contiguous()
        else:
            ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
            s = torch.tensor(ca / 127.0 if ca > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -ca, ca) / s), -127, 127).to(torch.int8)
            scales[name] = s
        quantized[name] = q.contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += q.numel() + (scales[name].numel() * scales[name].element_size())
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    pod = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dt = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].to(torch.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dt)
        else:
            out[name] = (q.float() * s.item()).to(dt)
    for name, t in obj["passthrough"].items():
        od = pod.get(name)
        out[name] = t.to(getattr(torch, od)) if isinstance(od, str) else t.detach().cpu()
    return out
