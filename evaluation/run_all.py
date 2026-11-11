"""python -m evaluation.run_all            full evaluation (a few minutes)
   python -m evaluation.run_all --quick    smoke run (seconds)
Writes evaluation/results/results.json and REPORT.md."""
import argparse

from evaluation.suite import run_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="evaluation/results")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--models", default=None, help="model directory (default: models)")
    a = ap.parse_args()
    run_all(a.quick, a.out, a.seed, a.models)
    print(f"done -> {a.out}/REPORT.md")


if __name__ == "__main__":
    main()
