import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from core.eval import build_sentencepiece_luts, eval_val


class FakeSentencePiece:
    """Hand-written stand-in for sentencepiece.SentencePieceProcessor.

    Implements only the methods build_sentencepiece_luts calls, over a vocab
    small enough that every LUT entry can be verified by hand.
    """

    # (piece, kind) per token id; kind in {"unknown", "control", "byte", "normal"}
    PIECES = [
        ("<unk>", "unknown"),        # 0: excluded from byte accounting
        ("<s>", "control"),          # 1: excluded
        ("<0x41>", "byte"),          # 2: byte-level token, 1 byte
        ("▁the", "normal"),     # 3: space-prefixed, "the" = 3 bytes
        ("cat", "normal"),           # 4: continuation piece, 3 bytes
        ("▁é", "normal"),  # 5: space-prefixed, "é" = 2 UTF-8 bytes
    ]

    def vocab_size(self):
        return len(self.PIECES)

    def is_unknown(self, tid):
        return self.PIECES[tid][1] == "unknown"

    def is_control(self, tid):
        return self.PIECES[tid][1] == "control"

    def is_unused(self, tid):
        return False

    def is_byte(self, tid):
        return self.PIECES[tid][1] == "byte"

    def id_to_piece(self, tid):
        return self.PIECES[tid][0]


class UniformModel(torch.nn.Module):
    """Predicts the uniform distribution, so per-token loss is exactly ln(V)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, target_ids):
        logits = torch.zeros(target_ids.numel(), self.vocab_size)
        return F.cross_entropy(logits, target_ids.reshape(-1))


# Model vocab is padded past the sp vocab (6 pieces -> 8 ids); ids 6-7 must be
# excluded like control tokens.
VOCAB = 8
DEVICE = torch.device("cpu")


def _args(**overrides):
    base = dict(val_batch_size=8, train_seq_len=4, val_max_tokens=0)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_lut_golden_values():
    bbl, hsl, ibl = build_sentencepiece_luts(FakeSentencePiece(), VOCAB, DEVICE)
    # bytes per target token: byte tokens count 1, "▁word" counts the word's
    # UTF-8 bytes (the leading space is accounted separately), control/unknown
    # count 0.
    assert bbl.tolist() == [0, 0, 1, 3, 3, 2, 0, 0]
    # space-prefix flag: only ▁-pieces.
    assert hsl.tolist() == [False, False, False, True, False, True, False, False]
    # excluded ids: unknown/control plus the padding beyond the sp vocab.
    assert ibl.tolist() == [True, True, False, False, False, False, True, True]


# One validation stream, two seq_len=4 rows:
#   x = [1, 3, 4, 3 | 2, 3, 5, 4]
#   y = [3, 4, 3, 2 | 3, 5, 4, 2]
# Byte count per target position (bbl[y] plus 1 leading-space byte when the
# target is ▁-prefixed AND the preceding token x is a real token, i.e. ~ibl[x]):
#   pos0 y=3: 3 bytes, no space (x=1 is control)        -> 3
#   pos1 y=4: 3 bytes                                   -> 3
#   pos2 y=3: 3 bytes + space (x=4 is real)             -> 4
#   pos3 y=2: 1 byte (byte token, never space-prefixed) -> 1
#   pos4 y=3: 3 bytes + space (x=2, byte tokens count as real) -> 4
#   pos5 y=5: 2 bytes + space (x=3 is real)             -> 3
#   pos6 y=4: 3 bytes                                   -> 3
#   pos7 y=2: 1 byte                                    -> 1
# total: 22 bytes over 8 tokens.
VAL_TOKENS = torch.tensor([1, 3, 4, 3, 2, 3, 5, 4, 2], dtype=torch.uint16)


def test_eval_val_bpb_golden():
    bbl, hsl, ibl = build_sentencepiece_luts(FakeSentencePiece(), VOCAB, DEVICE)
    vl, bpb = eval_val(_args(), UniformModel(VOCAB), 0, 1, DEVICE, 1,
                       VAL_TOKENS, bbl, hsl, ibl)
    assert vl == pytest.approx(math.log(VOCAB), rel=1e-6)
    # bpb = (loss / ln 2) * tokens / bytes = 3 bits/token * 8 tokens / 22 bytes
    assert bpb == pytest.approx(3.0 * 8 / 22, rel=1e-6)


def test_eval_val_respects_val_max_tokens():
    bbl, hsl, ibl = build_sentencepiece_luts(FakeSentencePiece(), VOCAB, DEVICE)
    # Cap at 4 tokens: only the first row is scored (3+3+4+1 = 11 bytes).
    vl, bpb = eval_val(_args(val_max_tokens=4), UniformModel(VOCAB), 0, 1,
                       DEVICE, 1, VAL_TOKENS, bbl, hsl, ibl)
    assert vl == pytest.approx(math.log(VOCAB), rel=1e-6)
    assert bpb == pytest.approx(3.0 * 4 / 11, rel=1e-6)


def test_eval_val_rejects_undersized_batch():
    bbl, hsl, ibl = build_sentencepiece_luts(FakeSentencePiece(), VOCAB, DEVICE)
    with pytest.raises(ValueError, match="VAL_BATCH_SIZE"):
        eval_val(_args(val_batch_size=2), UniformModel(VOCAB), 0, 1, DEVICE, 1,
                 VAL_TOKENS, bbl, hsl, ibl)
