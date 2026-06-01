#!/usr/bin/env python3
"""Observability toolkit for vocabulary-space models.

The premise of this project: because ``hidden_dim == vocab_size`` and there is
no embedding, every hidden state is already a distribution over words. The
computation is meant to be readable *by construction*. This tool reads it.

Subcommands
-----------
  trace      Top-k active vocab dims of the register state after each step,
             for a given input — watch a prediction take shape.
  wordmap    For v8_lowrank_vv: the learned word->word interaction matrix
             W = U @ V^T + diag(d). The clearest interpretability artifact:
             "which word activates which other word."
  causality  Perturb one vocab dimension of an intermediate register state and
             measure how much the model's predicted next-word distribution
             moves. Tests whether the readable state is mechanistically
             load-bearing or merely decorative — the make-or-break check.
  demo       Train v8 on a planted-bigram synthetic language on CPU, then show
             the wordmap *recovers* the planted transitions. A controlled
             faithfulness test (does reading the weights tell the truth?).

All commands run on CPU in float32. `trace`/`wordmap`/`causality` accept an
optional --checkpoint (a state_dict saved by train.py) and --version.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")  # repo root, so vN_*/ packages import
from core.config import Hyperparameters  # noqa: E402
from core.registry import build_model, get_registry  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_model(version: str, checkpoint: str | None):
    """Build a model on CPU (float32) and optionally load a checkpoint."""
    model = build_model(version, Hyperparameters())
    if checkpoint:
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)  # train.py ckpts wrap weights under "model"
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[load] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")
    return model.float().eval()


def step_modules(model):
    """Return [(label, module)] for the per-step register transforms."""
    for attr in ("hops", "steps", "layers", "blocks", "columns"):
        seq = getattr(model, attr, None)
        if isinstance(seq, torch.nn.ModuleList) and len(seq):
            return [(f"{attr}[{i}]", m) for i, m in enumerate(seq)]
    return []


def capture_register_states(model, input_ids, perturb=None):
    """Run a forward pass and capture the (B,T,V) state after each step.

    perturb: optional (step_index, position, dim, delta) — added to that step's
    output state before it flows on, for the causality probe.
    """
    V = model.vocab_size
    steps = step_modules(model)
    captured: list[torch.Tensor] = []
    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if perturb is not None:
                p_idx, pos, dim, delta = perturb
                if idx == p_idx:
                    o = o.clone()
                    o[:, pos, dim] = o[:, pos, dim] + delta
            captured.append(o.detach().float())
            return o
        return hook

    for i, (_label, mod) in enumerate(steps):
        handles.append(mod.register_forward_hook(make_hook(i)))
    target = torch.zeros_like(input_ids)
    with torch.no_grad():
        model(input_ids, target)
    for h in handles:
        h.remove()
    return [lbl for lbl, _ in steps], captured, V


def topk_words(vec, k, vocab=None):
    """Top-k indices of a 1-D activation vector, as (token, weight) strings."""
    vals, idx = torch.topk(vec, min(k, vec.numel()))
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        tok = vocab[i] if vocab and i < len(vocab) else str(i)
        out.append(f"{tok}({v:+.2f})")
    return out


def load_vocab(tokenizer):
    if not tokenizer:
        return None
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    return [sp.id_to_piece(i).replace("▁", "_") for i in range(sp.vocab_size())]


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_trace(a):
    model = load_model(a.version, a.checkpoint)
    vocab = load_vocab(a.tokenizer)
    ids = ([int(x) for x in a.prompt.split(",")] if a.prompt
           else torch.randint(0, model.vocab_size, (a.seqlen,)).tolist())
    input_ids = torch.tensor(ids).unsqueeze(0)
    pos = a.pos if a.pos >= 0 else len(ids) - 1
    labels, states, V = capture_register_states(model, input_ids)
    print(f"version={a.version} vocab={V} prompt_ids={ids} position={pos}\n")
    print(f"Top-{a.k} active vocab dims of the register state, position {pos}:")
    for lbl, st in zip(labels, states):
        words = topk_words(st[0, pos], a.k, vocab)
        print(f"  after {lbl:>10}: {'  '.join(words)}")
    print("\n(The last row is, up to the final norm/softcap, the predicted next-word distribution.)")


def cmd_wordmap(a):
    model = load_model(a.version, a.checkpoint)
    vocab = load_vocab(a.tokenizer)
    found = [(n, m) for n, m in model.named_modules()
             if hasattr(m, "U") and hasattr(m, "V") and hasattr(m, "diag")]
    if not found:
        print(f"{a.version} has no readable U@V^T+diag interaction (this is a v8-style artifact).")
        return
    print(f"version={a.version}: {len(found)} interaction map(s). Top word->word edges by |W|.\n")
    for name, m in found:
        W = (m.U.detach().float() @ m.V.detach().float().T)
        W = W + torch.diag(m.diag.detach().float())
        flat = W.abs().flatten()
        vals, idx = torch.topk(flat, a.k)
        print(f"[{name}]  (diag self-weight mean {m.diag.detach().float().mean():+.3f})")
        for v, fi in zip(vals.tolist(), idx.tolist()):
            i, j = divmod(fi, W.size(1))
            wi = vocab[i] if vocab and i < len(vocab) else f"#{i}"
            wj = vocab[j] if vocab and j < len(vocab) else f"#{j}"
            print(f"   {wi:>12} -> {wj:<12} W={W[i, j]:+.3f}")
        print()


def cmd_causality(a):
    model = load_model(a.version, a.checkpoint)
    ids = ([int(x) for x in a.prompt.split(",")] if a.prompt
           else torch.randint(0, model.vocab_size, (a.seqlen,)).tolist())
    input_ids = torch.tensor(ids).unsqueeze(0)
    pos = a.pos if a.pos >= 0 else len(ids) - 1
    labels, base_states, V = capture_register_states(model, input_ids)
    if not base_states:
        print("No step modules found to probe."); return
    p_step = a.step if a.step >= 0 else len(base_states) - 2  # perturb a mid state
    p_step = max(0, min(p_step, len(base_states) - 1))
    dim = a.dim if a.dim >= 0 else int(base_states[p_step][0, pos].argmax())
    base_pred = F.softmax(base_states[-1][0, pos], dim=-1)

    rows = []
    for delta in (a.delta, -a.delta):
        _, pert_states, _ = capture_register_states(
            model, input_ids, perturb=(p_step, pos, dim, delta))
        pert_pred = F.softmax(pert_states[-1][0, pos], dim=-1)
        kl = F.kl_div(pert_pred.clamp_min(1e-9).log(), base_pred, reduction="sum").item()
        top_changed = int(pert_pred.argmax()) != int(base_pred.argmax())
        rows.append((delta, kl, top_changed))
    print(f"version={a.version} vocab={V}")
    print(f"Perturbing register dim {dim} at step {labels[p_step]}, position {pos}, by ±{a.delta}:")
    print(f"  baseline top-1 prediction dim = {int(base_pred.argmax())} (p={base_pred.max():.3f})")
    for delta, kl, changed in rows:
        print(f"  delta={delta:+.2f}: KL(next-word dist shift)={kl:.4f}  top-1 changed={changed}")
    avg_kl = sum(r[1] for r in rows) / len(rows)
    verdict = ("LOAD-BEARING (readable state causally drives the prediction)"
               if avg_kl > a.threshold else
               "WEAK/DECORATIVE (perturbation barely moves the output)")
    print(f"\n  mean KL = {avg_kl:.4f}  ->  {verdict}")


def cmd_demo(a):
    """Train v8 on a planted bigram language, then check the map recovers it."""
    torch.manual_seed(0)
    V, hops, rank, seqlen, steps = 32, a.hops, 8, 24, a.steps
    cls = get_registry()["v8_lowrank_vv"]
    model = cls(vocab_size=V, num_hops=hops, interaction_rank=rank,
                activation="gelu", decay_init=3.0).float()

    # Planted ground truth: deterministic next-token map next[i] = perm[i].
    perm = torch.randperm(V)

    def batch(bs):
        starts = torch.randint(0, V, (bs, 1))
        seq = [starts]
        for _ in range(seqlen):
            seq.append(perm[seq[-1]])
        s = torch.cat(seq, dim=1)            # (bs, seqlen+1), each step follows perm
        return s[:, :-1], s[:, 1:]

    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    print(f"demo: training v8 (V={V}, hops={hops}, rank={rank}) on a planted bigram for {steps} steps...")
    for t in range(steps):
        x, y = batch(a.batch)
        loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % max(steps // 6, 1) == 0:
            print(f"  step {t+1:4d}  loss {loss.item():.4f}")

    # Read the LAST hop's word->word map and check argmax_j W[i,j] == perm[i].
    inter = model.hops[-1].interaction
    W = inter.U.detach() @ inter.V.detach().T + torch.diag(inter.diag.detach())
    pred_next = W.argmax(dim=1)
    recovery = (pred_next == perm).float().mean().item()
    print(f"\nPlanted: next[i] = perm[i]. Recovered from W = U@V^T+diag (last hop):")
    print(f"  exact word->word recovery (argmax_j W[i,j] == perm[i]): {recovery*100:.1f}%  (chance ≈ {100/V:.1f}%)")
    examples = [f"{i}->{int(pred_next[i])}(true {int(perm[i])})" for i in range(min(8, V))]
    print("  sample edges:", "  ".join(examples))
    verdict = "FAITHFUL: the readable map recovers the learned structure." if recovery > 0.5 \
        else "PARTIAL: map carries some structure; the full model spreads it across hops/propagation."
    print(f"\n  -> {verdict}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--version", default="v8_lowrank_vv")
        sp.add_argument("--checkpoint", default=None, help="state_dict from train.py")
        sp.add_argument("--tokenizer", default=None, help="sp model path to show words")

    t = sub.add_parser("trace"); common(t)
    t.add_argument("--prompt", default=None, help="comma-separated token ids")
    t.add_argument("--seqlen", type=int, default=8)
    t.add_argument("--pos", type=int, default=-1)
    t.add_argument("--k", type=int, default=6)
    t.set_defaults(func=cmd_trace)

    w = sub.add_parser("wordmap"); common(w)
    w.add_argument("--k", type=int, default=12)
    w.set_defaults(func=cmd_wordmap)

    c = sub.add_parser("causality"); common(c)
    c.add_argument("--prompt", default=None)
    c.add_argument("--seqlen", type=int, default=8)
    c.add_argument("--pos", type=int, default=-1)
    c.add_argument("--step", type=int, default=-1)
    c.add_argument("--dim", type=int, default=-1)
    c.add_argument("--delta", type=float, default=5.0)
    c.add_argument("--threshold", type=float, default=0.01)
    c.set_defaults(func=cmd_causality)

    d = sub.add_parser("demo")
    d.add_argument("--steps", type=int, default=400)
    d.add_argument("--batch", type=int, default=64)
    d.add_argument("--hops", type=int, default=1)
    d.set_defaults(func=cmd_demo)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
