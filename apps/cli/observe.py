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
    p_step = a.step if a.step >= 0 else max(0, len(base_states) - 2)  # a mid state
    p_step = max(0, min(p_step, len(base_states) - 1))
    mid = base_states[p_step][0, pos]
    # Perturb the dim the state actually has "on" (top-active word), unless told otherwise.
    dim = a.dim if a.dim >= 0 else int(mid.abs().argmax())
    scale = float(mid.abs().max())                  # local state magnitude (NOT a fixed delta)
    base_final = base_states[-1][0, pos]            # the register state IS the logits (pre-softcap)
    base_pred = F.softmax(base_final, dim=-1)

    # Interventions scaled to the state, so they bite regardless of magnitude:
    #   ablate  -> set the active word's dim to 0
    #   boost / suppress -> +/- k * (local magnitude)
    interventions = [
        ("ablate", -mid[dim].item()),
        ("boost", a.kscale * scale),
        ("suppress", -a.kscale * scale),
    ]
    print(f"version={a.version} vocab={V}  step={labels[p_step]} pos={pos} dim={dim}")
    print(f"  baseline top-1 = dim {int(base_pred.argmax())} (p={base_pred.max():.3f}); "
          f"|state|max at step = {scale:.1f}")
    best = 0.0
    for name, delta in interventions:
        _, ps, _ = capture_register_states(model, input_ids, perturb=(p_step, pos, dim, delta))
        pf = ps[-1][0, pos]
        # Logit-space change is the primary signal — softmax saturation hides post-softmax KL.
        rel = (pf - base_final).abs().sum().item() / (base_final.abs().sum().item() + 1e-9)
        pred = F.softmax(pf, dim=-1)
        kl = F.kl_div(pred.clamp_min(1e-9).log(), base_pred, reduction="sum").item()
        changed = int(pred.argmax()) != int(base_pred.argmax())
        best = max(best, rel)
        print(f"  {name:8} Δlogits(rel L1)={rel:.4f}  KL={kl:.4f}  top1_changed={changed}")
    verdict = ("LOAD-BEARING (perturbing the active word moves the output)"
               if best > a.threshold else
               "WEAK / not load-bearing at this site (output barely moves)")
    print(f"\n  max Δlogits(rel) = {best:.4f}  ->  {verdict}")


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


def cmd_sweep(a):
    """Map which (step, word) sites are causally load-bearing vs decorative."""
    model = load_model(a.version, a.checkpoint)
    vocab = load_vocab(a.tokenizer)
    ids = ([int(x) for x in a.prompt.split(",")] if a.prompt
           else torch.randint(0, model.vocab_size, (a.seqlen,)).tolist())
    input_ids = torch.tensor(ids).unsqueeze(0)
    pos = a.pos if a.pos >= 0 else len(ids) - 1
    labels, base, V = capture_register_states(model, input_ids)
    if not base:
        print("No step modules to sweep."); return
    base_final = base[-1][0, pos]
    base_arg = int(F.softmax(base_final, dim=-1).argmax())
    denom = base_final.abs().sum().item() + 1e-9

    print(f"version={a.version} vocab={V} pos={pos}  (Δlogits = rel. L1 change in pre-softcap logits)")
    print(f"  probing top-{a.topk} active word(s) per step; LOAD if Δlogits > {a.threshold}\n")
    print(f"  {'step':>10} {'word':>14} {'maxΔlogits':>11}")
    per_step_max = []
    for s in range(len(base)):
        mid = base[s][0, pos]
        scale = float(mid.abs().max())
        topdims = mid.abs().topk(min(a.topk, V)).indices.tolist()
        step_max = 0.0
        for dim in topdims:
            best_rel, changed = 0.0, False
            for delta in (-mid[dim].item(), a.kscale * scale, -a.kscale * scale):
                _, ps, _ = capture_register_states(model, input_ids, perturb=(s, pos, dim, delta))
                pf = ps[-1][0, pos]
                best_rel = max(best_rel, (pf - base_final).abs().sum().item() / denom)
                if int(F.softmax(pf, dim=-1).argmax()) != base_arg:
                    changed = True
            step_max = max(step_max, best_rel)
            word = vocab[dim] if vocab and dim < len(vocab) else f"#{dim}"
            flag = "LOAD" if best_rel > a.threshold else "."
            print(f"  {labels[s]:>10} {word:>14} {best_rel:>11.4f}  {flag}{' *flip*' if changed else ''}")
        per_step_max.append((labels[s], step_max))
    print("\n  per-step max Δlogits (where in depth the causal sites are):")
    for lbl, m in per_step_max:
        print(f"   {lbl:>10} {m:>8.4f} {'#' * min(int(m / max(a.threshold, 1e-9)), 40)}")
    load = [l for l, m in per_step_max if m > a.threshold]
    print(f"\n  load-bearing steps (Δlogits > {a.threshold}): {load or 'none at probed sites'}")


_COVERAGE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning the universe was created from nothing.",
    "Machine learning models predict the next token in a sequence.",
    "She walked into the room and quietly sat down by the window.",
    "Water freezes at zero degrees and boils at one hundred degrees.",
    "The president signed the bill into law on Tuesday afternoon.",
    "He opened the old wooden box and found a faded photograph inside.",
    "Researchers published their findings in a peer reviewed journal.",
    "The river flooded the village after three days of heavy rain.",
    "A small startup announced a new product at the conference today.",
    "Children laughed and played in the park until the sun went down.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Investors worried about rising interest rates and falling stocks.",
    "The ancient temple stood quietly at the edge of the desert.",
]


def cmd_coverage(a):
    """Faithfulness coverage: the fraction of readable active-word sites that are
    causally load-bearing, averaged over real-text prompts. One number for the
    observability axis, to track against bits-per-byte (the performance axis)."""
    torch.manual_seed(0)
    model = load_model(a.version, a.checkpoint)
    if a.tokenizer:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
        prompts = [sp.encode(s, out_type=int) for s in _COVERAGE_SENTENCES]
    else:  # fallback: out-of-distribution random prompts
        prompts = [torch.randint(0, model.vocab_size, (16,)).tolist() for _ in range(5)]
    prompts = [p for p in prompts if len(p) >= 4]

    import statistics
    rels = []                 # per-site best Δlogits across all prompts/positions
    per_step_rels = {}
    for ids in prompts:
        input_ids = torch.tensor(ids).unsqueeze(0)
        labels, base, V = capture_register_states(model, input_ids)
        if not base:
            continue
        L = len(ids)
        for pos in sorted({max(L // 3, 1), 2 * L // 3, L - 1}):
            base_final = base[-1][0, pos]
            denom = base_final.abs().sum().item() + 1e-9
            for s in range(len(base)):
                mid = base[s][0, pos]
                scale = float(mid.abs().max())
                for dim in mid.abs().topk(min(a.topk, V)).indices.tolist():
                    best = 0.0
                    for delta in (-mid[dim].item(), a.kscale * scale, -a.kscale * scale):
                        _, ps, _ = capture_register_states(model, input_ids, perturb=(s, pos, dim, delta))
                        best = max(best, (ps[-1][0, pos] - base_final).abs().sum().item() / denom)
                    rels.append(best)
                    per_step_rels.setdefault(labels[s], []).append(best)

    n = len(rels)
    print(f"version={a.version}  prompts={len(prompts)}  sites_probed={n}")
    print(f"  mean Δlogits = {sum(rels) / max(n, 1):.4f}   median = {statistics.median(rels) if rels else 0:.4f}")
    print("  FAITHFULNESS COVERAGE (fraction of sites with Δlogits > τ) — robustness to τ:")
    for t in (0.01, 0.02, 0.05, 0.10):
        cov = sum(1 for r in rels if r > t) / max(n, 1)
        print(f"     τ={t:<5}: {cov * 100:5.1f}%")
    print(f"  per-step coverage (τ={a.threshold}):")
    for lbl, rs in per_step_rels.items():
        c = sum(1 for r in rs if r > a.threshold) / max(len(rs), 1)
        print(f"   {lbl:>10}: {100 * c:3.0f}%  (mean Δ {sum(rs) / max(len(rs), 1):.3f})")


def cmd_induction(a):
    """Beyond-bigram test: in-context key->value association (induction).

    Each sequence carries its OWN random key->value map: a study phase presents
    (k, v) pairs, then a query phase re-presents each key and the model must
    predict its value. A fixed-weight bigram is at chance (1/V) on query
    positions — only in-context lookback can solve it. Train v8, measure
    query-position accuracy, then read the mechanism with a trace."""
    from types import SimpleNamespace
    torch.manual_seed(0)
    V, P, hops, rank = a.vocab, a.pairs, a.hops, a.rank
    # Small-but-real args so any variant builds (mirrors tests/_SMALL_ARGS).
    ns = SimpleNamespace(
        vocab_size=V, num_steps=hops, n_channels=64, n_fourier_basis=16,
        logit_softcap=30.0, decay_init=3.0, activation="gelu",
        num_heads=2, num_kv_heads=2, rope_base=10000.0, qk_gain_init=1.5, kernel_size=4,
        unique_steps=hops, invocations_per_step=1, n_heads=2, transform_rank=8,
        band_split="1,1,2", slow_decay_init=4.0, fast_decay_init=2.0,
        n_ops=4, interaction_rank=rank, state_dim=32, inner_dim=64,
        k_active=16, inner_mul=2, parallel_waves=True, grad_checkpoint=False,
        embed_dim=32, gumbel_tau=1.0, halt_threshold=0.5, ponder_lambda=0.01,
        sparsity_k=8, aux_loss_weight=0.1, aux_loss_decay=0.9,
        num_columns=2, steps_per_column=2, n_branches=2,
    )
    model = build_model(a.version, ns).float()

    def make_batch(bs):
        xs, ys, masks = [], [], []
        for _ in range(bs):
            keys = torch.randperm(V)[:P]
            vals = torch.randint(0, V, (P,))
            toks, qvpos = [], []
            for i in range(P):                       # study: k, v, k, v, ...
                toks += [int(keys[i]), int(vals[i])]
            for i in torch.randperm(P).tolist():     # query: k, <predict v>
                toks.append(int(keys[i]))
                qvpos.append(len(toks))              # index in toks of the value
                toks.append(int(vals[i]))
            seq = torch.tensor(toks)
            xs.append(seq[:-1]); ys.append(seq[1:])
            m = torch.zeros(seq.numel() - 1, dtype=torch.bool)
            for qp in qvpos:
                if qp - 1 < m.numel():
                    m[qp - 1] = True                 # query-key position predicts the value
            masks.append(m)
        return torch.stack(xs), torch.stack(ys), torch.stack(masks)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"induction: {a.version} (V={V}, pairs={P}, steps_dim={hops}, rank={rank}), {a.steps} steps; chance = {100/V:.1f}%")
    for t in range(a.steps):
        x, y, _ = make_batch(a.batch)
        loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % max(a.steps // 8, 1) == 0:
            print(f"  step {t+1:4d}  full-seq loss {loss.item():.4f}")

    x, y, mask = make_batch(512)
    _, states, _ = capture_register_states(model, x)   # final state = logits up to monotone ops
    if not states:
        print(f"\n  cannot probe {a.version}: no per-step ModuleList to hook "
              f"(tool supports hops/steps/layers/blocks/columns; weight-shared loops unsupported).")
        return
    pred = states[-1].argmax(-1)
    total = int(mask.sum())
    acc = int(((pred == y) & mask).sum()) / max(total, 1)
    print(f"\n  query-position accuracy = {acc*100:.1f}%   (chance = bigram ceiling = {100/V:.1f}%)")
    print(f"  -> {'SOLVES INDUCTION — in-context lookback works (beyond bigram)' if acc > 5.0/V else 'at/near chance — no in-context lookback'}")

    x1, y1, m1 = make_batch(1)
    qpos = int(m1[0].nonzero()[0])
    _, st, _ = capture_register_states(model, x1)
    truth = int(y1[0, qpos])
    print(f"\n  trace @ query position {qpos} (true recalled value = token {truth}):")
    for i, s in enumerate(st):
        v, idx = torch.topk(s[0, qpos], 4)
        toks = "  ".join(f"{int(j)}({float(w):+.1f})" for w, j in zip(v, idx))
        hit = "  <-- recalled value on top" if int(idx[0]) == truth else ""
        print(f"    hops[{i}]: {toks}{hit}")


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
    c.add_argument("--kscale", type=float, default=3.0, help="boost/suppress = k * local state magnitude")
    c.add_argument("--threshold", type=float, default=0.02, help="rel. logit change to call it load-bearing")
    c.set_defaults(func=cmd_causality)

    d = sub.add_parser("demo")
    d.add_argument("--steps", type=int, default=400)
    d.add_argument("--batch", type=int, default=64)
    d.add_argument("--hops", type=int, default=1)
    d.set_defaults(func=cmd_demo)

    sw = sub.add_parser("sweep"); common(sw)
    sw.add_argument("--prompt", default=None)
    sw.add_argument("--seqlen", type=int, default=8)
    sw.add_argument("--pos", type=int, default=-1)
    sw.add_argument("--topk", type=int, default=3, help="top active dims to probe per step")
    sw.add_argument("--kscale", type=float, default=3.0)
    sw.add_argument("--threshold", type=float, default=0.02)
    sw.set_defaults(func=cmd_sweep)

    cv = sub.add_parser("coverage"); common(cv)
    cv.add_argument("--topk", type=int, default=2, help="top active dims probed per step")
    cv.add_argument("--kscale", type=float, default=3.0)
    cv.add_argument("--threshold", type=float, default=0.02)
    cv.set_defaults(func=cmd_coverage)

    ind = sub.add_parser("induction")
    ind.add_argument("--version", default="v8_lowrank_vv")
    ind.add_argument("--vocab", type=int, default=48)
    ind.add_argument("--pairs", type=int, default=10)
    ind.add_argument("--hops", type=int, default=8)
    ind.add_argument("--rank", type=int, default=32)
    ind.add_argument("--steps", type=int, default=2000)
    ind.add_argument("--batch", type=int, default=64)
    ind.add_argument("--lr", type=float, default=0.01)
    ind.set_defaults(func=cmd_induction)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
