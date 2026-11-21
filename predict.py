"""CLI for the Kazakh profanity classifier.

    python predict.py "comment text"     -> one prediction
    python predict.py                    -> interactive; empty line quits
"""
import sys
from src.kkprof import load, predict


def show(text, bundle):
    r = predict(text, bundle)
    verdict = "PROFANITY (1)" if r["label"] else "no profanity (0)"
    print(f"{verdict}  prob={r['prob']:.4f}  threshold={r['threshold']:.3f}")


if __name__ == "__main__":
    b = load()
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]), b)
    else:
        while True:
            t = input("comment> ").strip()
            if not t:
                break
            show(t, b)
