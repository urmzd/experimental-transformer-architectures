"""
Download Python code from The Stack, train a 1024-token BPE tokenizer,
and produce binary shards in the same format as FineWeb.

Usage:
    python data/prepare_code.py [--max-docs 100000] [--val-ratio 0.02]

Produces:
    data/tokenizers/code_1024_bpe.model
    data/datasets/code_sp1024/code_train_000000.bin ...
    data/datasets/code_sp1024/code_val_000000.bin ...
"""

import argparse
import io
import struct
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets" / "code_sp1024"
TOKENIZERS_DIR = ROOT / "tokenizers"


def download_code(max_docs: int) -> list[str]:
    """Download Python files from The Stack v2 (smol subset)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"Downloading up to {max_docs} Python files from The Stack...")
    ds = load_dataset(
        "bigcode/starcoderdata",
        data_dir="python",
        split="train",
        streaming=True,
    )

    docs = []
    for i, example in enumerate(ds):
        if i >= max_docs:
            break
        content = example.get("content", "")
        if len(content) > 100:  # skip tiny files
            docs.append(content)
        if (i + 1) % 10000 == 0:
            print(f"  downloaded {i + 1} docs ({len(docs)} kept)")

    print(f"Downloaded {len(docs)} Python files")
    return docs


def train_tokenizer(docs: list[str], vocab_size: int = 1024) -> str:
    """Train a SentencePiece BPE tokenizer on the code corpus."""
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("pip install sentencepiece")

    TOKENIZERS_DIR.mkdir(parents=True, exist_ok=True)
    model_prefix = str(TOKENIZERS_DIR / "code_1024_bpe")

    # Write docs to temp file for sentencepiece
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for doc in docs:
            # SentencePiece expects one sentence per line
            for line in doc.split("\n"):
                line = line.strip()
                if line:
                    f.write(line + "\n")
        temp_path = f.name

    print(f"Training {vocab_size}-token BPE tokenizer...")
    spm.SentencePieceTrainer.train(
        input=temp_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        max_sentence_length=8192,
        input_sentence_size=min(len(docs) * 10, 5_000_000),
        shuffle_input_sentence=True,
    )

    Path(temp_path).unlink()
    model_path = model_prefix + ".model"
    print(f"Tokenizer saved to {model_path}")
    return model_path


def tokenize_and_shard(
    docs: list[str],
    tokenizer_path: str,
    val_ratio: float = 0.02,
    tokens_per_shard: int = 10_000_000,
):
    """Tokenize docs and write binary shards (uint16)."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    vocab_size = sp.vocab_size()
    assert vocab_size <= 65535, "vocab too large for uint16"

    print(f"Tokenizing {len(docs)} docs with {vocab_size}-token vocabulary...")

    # Tokenize all docs
    all_tokens = []
    for i, doc in enumerate(docs):
        tokens = sp.encode(doc, out_type=int)
        all_tokens.extend(tokens)
        if (i + 1) % 10000 == 0:
            print(f"  tokenized {i + 1}/{len(docs)} docs ({len(all_tokens)} tokens)")

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens)}")

    # Split train/val
    n_val = max(int(len(all_tokens) * val_ratio), tokens_per_shard)
    val_tokens = all_tokens[:n_val]
    train_tokens = all_tokens[n_val:]

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Write val shards
    n_val_shards = max(1, len(val_tokens) // tokens_per_shard)
    for i in range(n_val_shards):
        start = i * tokens_per_shard
        end = min(start + tokens_per_shard, len(val_tokens))
        shard = val_tokens[start:end]
        path = DATASETS_DIR / f"code_val_{i:06d}.bin"
        shard.tofile(str(path))
        print(f"  wrote {path} ({len(shard)} tokens)")

    # Write train shards
    n_train_shards = max(1, len(train_tokens) // tokens_per_shard)
    for i in range(n_train_shards):
        start = i * tokens_per_shard
        end = min(start + tokens_per_shard, len(train_tokens))
        shard = train_tokens[start:end]
        path = DATASETS_DIR / f"code_train_{i:06d}.bin"
        shard.tofile(str(path))
        print(f"  wrote {path} ({len(shard)} tokens)")

    print(f"Done: {n_train_shards} train shards, {n_val_shards} val shards")
    print(f"  train: {len(train_tokens)} tokens")
    print(f"  val: {len(val_tokens)} tokens")


def main():
    parser = argparse.ArgumentParser(description="Prepare code dataset for training")
    parser.add_argument("--max-docs", type=int, default=100_000,
                        help="Max Python files to download (default: 100K)")
    parser.add_argument("--val-ratio", type=float, default=0.02,
                        help="Fraction of tokens for validation (default: 0.02)")
    parser.add_argument("--vocab-size", type=int, default=1024,
                        help="Tokenizer vocabulary size (default: 1024)")
    args = parser.parse_args()

    docs = download_code(args.max_docs)
    tokenizer_path = train_tokenizer(docs, args.vocab_size)
    tokenize_and_shard(docs, tokenizer_path, args.val_ratio)

    print("\nTo train on code:")
    print("  DATA_PATH=./data/datasets/code_sp1024 \\")
    print("  TOKENIZER_PATH=./data/tokenizers/code_1024_bpe.model \\")
    print("  MODEL_VERSION=v8_graph \\")
    print("  torchrun --standalone --nproc_per_node=3 train.py")


if __name__ == "__main__":
    main()
