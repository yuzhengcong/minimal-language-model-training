import cProfile
import pstats
import tempfile
import os
from adapters import run_train_bpe, _merge_pair_frequencies

corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

special_token = ["<|endoftext|>"]

# Write corpus to a temp file since run_train_bpe expects a file path
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
    f.write(corpus)
    tmp_path = f.name

try:
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = run_train_bpe(tmp_path, 263, special_token)

    profiler.disable()

    print("=== Merges (in order) ===")
    for i, (t1, t2) in enumerate(merges):
        # Merge elements may be ints or bytes; normalize to bytes for display
        b1 = bytes([t1]) if isinstance(t1, int) else t1
        b2 = bytes([t2]) if isinstance(t2, int) else t2
        s1 = b1.decode("utf-8", errors="replace")
        s2 = b2.decode("utf-8", errors="replace")
        s3 = (b1 + b2).decode("utf-8", errors="replace")
        print(f"  {i+1}: '{s1}' + '{s2}' -> '{s3}'")

    print(f"\n=== Vocab ({len(vocab)} tokens) ===")
    for tid, tbytes in sorted(vocab.items()):
        if tid >= 256:  # only show non-byte tokens
            if isinstance(tbytes, int):
                tbytes = bytes([tbytes])
            s = tbytes.decode("utf-8", errors="replace")
            print(f"  {tid}: '{s}'")
    print(merges)

    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumulative")
    # print("\n=== Top 20 by CUMULATIVE time ===")
    # stats.print_stats(20)

    # stats.sort_stats("tottime")
    # print("\n=== Top 20 by TOTAL (self) time ===")
    # stats.print_stats(20)
finally:
    os.unlink(tmp_path)


