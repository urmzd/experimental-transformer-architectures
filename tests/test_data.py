import numpy as np
import pytest
import torch

from glassbox_lm.core.data import (DistributedTokenLoader, TokenStream, load_data_shard,
                       load_validation_tokens)

MAGIC, FORMAT_VERSION = 20240520, 1


def write_shard(path, tokens):
    header = np.zeros(256, dtype="<i4")
    header[0], header[1], header[2] = MAGIC, FORMAT_VERSION, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, dtype="<u2").tobytes())


def u16_arange(start, stop):
    # torch.arange has no uint16 CPU kernel; go through numpy.
    return torch.from_numpy(np.arange(start, stop, dtype=np.uint16))


def test_load_data_shard_roundtrip(tmp_path):
    tokens = np.arange(100, dtype=np.uint16)
    path = tmp_path / "shard_000.bin"
    write_shard(path, tokens)
    loaded = load_data_shard(path)
    assert loaded.dtype == torch.uint16
    assert torch.equal(loaded, u16_arange(0, 100))


def test_load_data_shard_rejects_bad_magic(tmp_path):
    path = tmp_path / "shard_000.bin"
    write_shard(path, np.arange(10, dtype=np.uint16))
    raw = bytearray(path.read_bytes())
    raw[0:4] = np.array([12345], dtype="<i4").tobytes()
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="Bad header"):
        load_data_shard(path)


def test_load_validation_tokens_trims_to_seq_len_multiple(tmp_path):
    write_shard(tmp_path / "val_000.bin", np.arange(10, dtype=np.uint16))
    # 10 tokens, seq_len 4 -> 2 full (x, y) sequences need 8 + 1 tokens.
    tokens = load_validation_tokens(str(tmp_path / "val_*.bin"), seq_len=4)
    assert torch.equal(tokens, u16_arange(0, 9))


def test_token_stream_crosses_shard_boundary(tmp_path):
    # Distinct value ranges per shard so positions are recoverable from values.
    write_shard(tmp_path / "shard_000.bin", np.arange(0, 12, dtype=np.uint16))
    write_shard(tmp_path / "shard_001.bin", np.arange(100, 112, dtype=np.uint16))
    stream = TokenStream(str(tmp_path / "shard_*.bin"))
    took = stream.take(20)
    expected = torch.cat([u16_arange(0, 12), u16_arange(100, 108)])
    assert torch.equal(took, expected)
    assert torch.equal(stream.take(4), u16_arange(108, 112))


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_loader_ranks_disjoint_and_in_order(tmp_path, world_size):
    """Per-rank batches must partition the token stream with no overlap and no
    replay across successive batches.

    Each rank r takes span = local_tokens + 1 consecutive tokens starting at
    r * span; the +1 overlap exists only within a rank (y is x shifted by one),
    never across ranks.
    """
    seq_len, gas = 8, 1
    global_tokens = world_size * 2 * seq_len
    local_tokens = global_tokens // (world_size * gas)
    span = local_tokens + 1
    write_shard(tmp_path / "train_000.bin",
                np.arange(4 * world_size * span, dtype=np.uint16))
    pattern = str(tmp_path / "train_*.bin")
    device = torch.device("cpu")

    loaders = [DistributedTokenLoader(pattern, r, world_size, device)
               for r in range(world_size)]
    for batch_idx in range(2):
        base = batch_idx * world_size * span
        seen = []
        for r, loader in enumerate(loaders):
            x, y = loader.next_batch(global_tokens, seq_len, gas)
            assert x.shape == (local_tokens // seq_len, seq_len)
            assert torch.equal(y.reshape(-1)[:-1], x.reshape(-1)[1:])
            start = base + r * span
            assert torch.equal(x.reshape(-1),
                               torch.arange(start, start + local_tokens))
            seen.append(x.reshape(-1))
        flat = torch.cat(seen)
        assert flat.unique().numel() == flat.numel(), "ranks overlap"
