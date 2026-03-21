from __future__ import annotations

import math

import numpy as np
import torch
import torch.distributed as dist


def build_sentencepiece_luts(sp, vocab_size, device):
    sv = int(sp.vocab_size())
    ts = max(sv, vocab_size)
    bb = np.zeros((ts,), dtype=np.int16)
    hs = np.zeros((ts,), dtype=np.bool_)
    ib = np.ones((ts,), dtype=np.bool_)
    for tid in range(sv):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        ib[tid] = False
        if sp.is_byte(tid):
            bb[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            hs[tid] = True
            piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hs, dtype=torch.bool, device=device),
            torch.tensor(ib, dtype=torch.bool, device=device))


def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bbl, hsl, ibl):
    lbt = args.val_batch_size // (world_size * grad_accum_steps)
    if lbt < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE too small")
    lbs = lbt // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    ss, se = (total_seqs * rank) // world_size, (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(ss, se, lbs):
            be = min(bs + lbs, se)
            local = val_tokens[bs * args.train_seq_len:be * args.train_seq_len + 1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            ct = float(y.numel())
            loss_sum += bl.to(torch.float64) * ct
            tok_count += ct
            tb = bbl[y.reshape(-1)].to(torch.float64)
            tb += (hsl[y.reshape(-1)] & ~ibl[x.reshape(-1)]).to(torch.float64)
            byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_count).item()
    model.train()
    return vl, vl / math.log(2.0) * (tok_count.item() / byte_count.item())
