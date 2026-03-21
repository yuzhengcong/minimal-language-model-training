"""Profile BPE training using cProfile + pstats."""
import cProfile
import pstats
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tests"))

INPUT_PATH = Path(__file__).parent / "tests" / "fixtures" / "corpus.en"
VOCAB_SIZE = 500
SPECIAL_TOKENS = ["<|endoftext|>"]

if __name__ == "__main__":
    from adapters import run_train_bpe

    pr = cProfile.Profile()
    pr.enable()
    run_train_bpe(input_path=INPUT_PATH, vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    print("\n--- sorted by tottime (self time) ---")
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(20)
    print(s2.getvalue())
