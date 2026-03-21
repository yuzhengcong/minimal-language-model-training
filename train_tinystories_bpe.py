"""Train BPE on TinyStories, serialize vocab/merges, report stats."""
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tests"))

INPUT_PATH = Path(__file__).parent / "data" / "TinyStoriesV2-GPT4-train.txt"
OUTPUT_DIR = Path(__file__).parent / "data"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]

if __name__ == "__main__":
    from adapters import run_train_bpe

    tracemalloc.start()
    t0 = time.time()

    vocab, merges = run_train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    elapsed = time.time() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Serialize vocab: {id: hex string of bytes}
    vocab_serializable = {k: v.hex() for k, v in vocab.items()}
    with open(OUTPUT_DIR / "tinystories_vocab.json", "w") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

    # Serialize merges: list of [hex1, hex2]
    merges_serializable = [[a.hex(), b.hex()] for a, b in merges]
    with open(OUTPUT_DIR / "tinystories_merges.json", "w") as f:
        json.dump(merges_serializable, f, ensure_ascii=False, indent=2)

    # Stats
    print(f"\n=== Training Stats ===")
    print(f"Time:        {elapsed:.1f}s ({elapsed/3600:.4f} hours)")
    print(f"Peak memory: {peak_bytes / 1024**3:.2f} GB")
    print(f"Vocab size:  {len(vocab)}")
    print(f"Merges:      {len(merges)}")

    # Longest token
    longest_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_bytes = vocab[longest_id]
    print(f"\n=== Longest Token ===")
    print(f"Token ID:    {longest_id}")
    print(f"Length:      {len(longest_bytes)} bytes")
    try:
        print(f"Decoded:     {longest_bytes.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print(f"Hex:         {longest_bytes.hex()}")

    # Top 5 longest tokens
    print("\n=== Top 5 Longest Tokens ===")
    sorted_by_len = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for tid, tb in sorted_by_len:
        try:
            print(f"  [{tid}] len={len(tb):3d}  {tb.decode('utf-8')!r}")
        except UnicodeDecodeError:
            print(f"  [{tid}] len={len(tb):3d}  (hex) {tb.hex()}")
