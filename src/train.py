"""
Ablation experiments c1–c4: train NiSNN-A variants under 5-fold cross-validation
on the control and stress datasets, then print a results table.

Usage:
    .venv/bin/python src/train.py                          # all 6 conditions
    .venv/bin/python src/train.py --variants full          # c1/c2 only
    .venv/bin/python src/train.py --datasets control       # control only
    .venv/bin/python src/train.py --epochs 200             # longer run
    .venv/bin/python src/train.py --out my_results.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

# Ensure src/ is on path when running from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import build_model  # noqa: E402

try:
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    raise SystemExit("scikit-learn is required: pip install scikit-learn")

# ── constants ─────────────────────────────────────────────────────────────────

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
N_FOLDS = 5
EPOCHS  = 100
BATCH   = 32
LR      = 1e-3
SEED    = 42


# ── data loading ──────────────────────────────────────────────────────────────

def load_dataset(variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and per-trial z-score a dataset variant."""
    base = pathlib.Path("data") / variant
    X = np.load(base / f"{variant}_trials.npy")   # (288, 20, 20, 20) float32
    y = np.load(base / f"{variant}_labels.npy")   # (288,) int64

    # Per-trial, per-channel z-score over (S, T) dims
    mean = X.mean(axis=(-2, -1), keepdims=True)
    std  = X.std( axis=(-2, -1), keepdims=True).clip(1e-8)
    X = (X - mean) / std

    return torch.from_numpy(X), torch.from_numpy(y)


# ── training / evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_acc(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        correct += (model(X_batch).argmax(1) == y_batch).sum().item()
        total   += len(y_batch)
    return correct / total


# ── cross-validation ──────────────────────────────────────────────────────────

def run_cv(model_variant: str, data_variant: str, epochs: int) -> list[float]:
    torch.manual_seed(SEED)
    X, y = load_dataset(data_variant)
    dataset   = TensorDataset(X, y)
    labels_np = y.numpy()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(labels_np, labels_np)):
        torch.manual_seed(SEED + fold)
        model     = build_model(model_variant).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=BATCH, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=BATCH, shuffle=False
        )

        best_acc = 0.0
        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            acc = eval_acc(model, val_loader, DEVICE)
            if acc > best_acc:
                best_acc = acc

        fold_accs.append(best_acc)
        print(
            f"  [{model_variant}/{data_variant}] fold {fold + 1}/{N_FOLDS}: "
            f"{best_acc * 100:.1f}%"
        )

    return fold_accs


# ── main ──────────────────────────────────────────────────────────────────────

ALL_MODELS   = ["full", "encoder_only", "fixed_encoder"]
ALL_DATASETS = ["control", "stress"]


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments c1–c4")
    parser.add_argument(
        "--variants", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS, metavar="VARIANT",
        help="Model variants to run (default: all three)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS, metavar="DATASET",
        help="Dataset variants to run (default: control and stress)"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out",    type=str, default="results.json")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Epochs per fold: {args.epochs}  |  Folds: {N_FOLDS}  |  Batch: {BATCH}\n")

    results: dict[str, dict] = {}

    for model_v in args.variants:
        for data_v in args.datasets:
            key  = f"{model_v}/{data_v}"
            accs = run_cv(model_v, data_v, args.epochs)
            results[key] = {
                "fold_accs": accs,
                "mean":      float(np.mean(accs)),
                "std":       float(np.std(accs)),
            }

    # ── results table ─────────────────────────────────────────────────────────
    col = 62
    print()
    print("=" * col)
    print(f"{'Model':<18} {'Dataset':<10} {'Mean acc':>10} {'±Std':>7}   Folds")
    print("-" * col)
    for key, r in results.items():
        model_v, data_v = key.split("/")
        fold_str = " ".join(f"{a * 100:.1f}" for a in r["fold_accs"])
        print(
            f"{model_v:<18} {data_v:<10} {r['mean'] * 100:>9.1f}%"
            f" {r['std'] * 100:>6.1f}%   [{fold_str}]"
        )
    print("=" * col)
    print()

    # ── c1 pass/fail ──────────────────────────────────────────────────────────
    c1 = results.get("full/control")
    if c1 is not None:
        if c1["mean"] > 0.55:
            print(
                f"c1 PASS: full model beats chance on control "
                f"({c1['mean'] * 100:.1f}% > 55%)"
            )
        else:
            print(
                f"c1 FAIL: full model at {c1['mean'] * 100:.1f}% — "
                "check dataset loading or training stability"
            )
        print()

    # ── hypothesis verdict ────────────────────────────────────────────────────
    enc   = results.get("encoder_only/control")
    fixed = results.get("fixed_encoder/control")
    full  = results.get("full/control")
    if enc and full:
        gap = full["mean"] - enc["mean"]
        if gap < 0.05:
            print(
                f"Hypothesis SUPPORTED (control): encoder-only matches full model "
                f"(gap = {gap * 100:.1f}pp). Discriminative work is in the encoder."
            )
        else:
            print(
                f"Hypothesis NOT SUPPORTED (control): encoder-only is "
                f"{gap * 100:.1f}pp below full model. Classifier layers contribute."
            )
    if fixed and full:
        gap = full["mean"] - fixed["mean"]
        print(
            f"c4 (fixed encoder vs full, control): {gap * 100:.1f}pp gap — "
            + ("learned encoder was essential." if gap > 0.10 else "fixed encoder nearly sufficient.")
        )

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
