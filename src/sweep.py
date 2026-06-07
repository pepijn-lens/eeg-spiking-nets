"""
Dataset-size sweep: generate datasets at multiple trial counts,
run all ablation experiments for each, and save accumulated results.

Usage:
    .venv/bin/python src/sweep.py                              # full sweep
    .venv/bin/python src/sweep.py --sizes 72 288               # subset of sizes
    .venv/bin/python src/sweep.py --variants full encoder_only # subset of models
    .venv/bin/python src/sweep.py --epochs 200 --out sweep_results.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from generate import generate_dataset, save_dataset
from train import run_cv

DEFAULT_SIZES = [72, 144, 288, 576, 1152]
ALL_MODELS    = ["full", "encoder_only", "fixed_encoder"]
BASE_VARIANTS = ["control", "stress"]

VARIANT_DEFAULTS = {
    "control": dict(modulation_depth=0.7,  noise_level=1.0),
    "stress":  dict(modulation_depth=0.15, noise_level=3.0),
}
SEED = 42


def _variant_name(base: str, n: int) -> str:
    return f"{base}_N{n}"


def ensure_dataset(base_variant: str, n: int) -> str:
    """Generate dataset for base_variant at trial count n if not already on disk."""
    name    = _variant_name(base_variant, n)
    out_dir = pathlib.Path("data") / name
    if (out_dir / f"{name}_trials.npy").exists():
        print(f"  [skip generate] {name} already on disk")
        return name
    params = VARIANT_DEFAULTS[base_variant]
    data   = generate_dataset(n_trials=n, seed=SEED, **params)
    save_dataset(data, out_dir, name)
    return name


def main():
    parser = argparse.ArgumentParser(description="Sweep ablation experiments across dataset sizes")
    parser.add_argument("--sizes",    nargs="+", type=int, default=DEFAULT_SIZES, metavar="N",
                        help="Trial counts to sweep (default: 72 144 288 576 1152)")
    parser.add_argument("--variants", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="VARIANT")
    parser.add_argument("--epochs",   type=int,  default=100)
    parser.add_argument("--out",      type=str,  default="sweep_results.json")
    args = parser.parse_args()

    # results[model_variant][base_variant][str(N)] = {fold_accs, mean, std}
    results: dict = {m: {bv: {} for bv in BASE_VARIANTS} for m in args.variants}

    for n in sorted(args.sizes):
        print(f"\n{'='*58}")
        print(f"N = {n} trials")
        print("=" * 58)
        for base_variant in BASE_VARIANTS:
            variant_name = ensure_dataset(base_variant, n)
            for model_v in args.variants:
                print(f"\n  {model_v} / {base_variant} / N={n}")
                fold_accs = run_cv(model_v, variant_name, args.epochs)
                results[model_v][base_variant][str(n)] = {
                    "fold_accs": fold_accs,
                    "mean":      float(np.mean(fold_accs)),
                    "std":       float(np.std(fold_accs)),
                }

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.out}")

    col = 64
    print()
    print("=" * col)
    print(f"{'Model':<18} {'Dataset':<10} {'N':>6} {'Mean acc':>10} {'±Std':>7}")
    print("-" * col)
    for model_v in args.variants:
        for bv in BASE_VARIANTS:
            for n_str, r in sorted(results[model_v][bv].items(), key=lambda kv: int(kv[0])):
                print(
                    f"{model_v:<18} {bv:<10} {n_str:>6} "
                    f"{r['mean'] * 100:>9.1f}% {r['std'] * 100:>6.1f}%"
                )
    print("=" * col)


if __name__ == "__main__":
    main()
