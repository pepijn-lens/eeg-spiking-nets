"""
Visualise sweep results: accuracy vs. dataset size for all model variants.

Usage:
    .venv/bin/python src/plot_sweep.py
    .venv/bin/python src/plot_sweep.py --in sweep_results.json --out figures/fig_sweep.png
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt

MODEL_LABELS = {
    "full":          "Full NiSNN-A (c1/c2)",
    "encoder_only":  "Encoder only (c3)",
    "fixed_encoder": "Fixed encoder (c4)",
}
COLOURS = {
    "full":          "#1f77b4",
    "encoder_only":  "#2ca02c",
    "fixed_encoder": "#d62728",
}
DATASET_TITLES = {
    "control": "Control  (70% ERD)",
    "stress":  "Stress   (15% ERD)",
}


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results from sweep_results.json")
    parser.add_argument("--in",  dest="src", default="sweep_results.json")
    parser.add_argument("--out", dest="dst", default="figures/fig_sweep.png")
    args = parser.parse_args()

    src = pathlib.Path(args.src)
    if not src.exists():
        raise SystemExit(f"Results file not found: {src}  —  run sweep.py first")

    data    = json.loads(src.read_text())
    models  = list(data.keys())
    bvs     = ["control", "stress"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    fig.suptitle("Ablation accuracy vs. dataset size  (5-fold CV, ±1 std)", fontsize=13)

    for ax, bv in zip(axes, bvs):
        ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, label="Chance (50%)", zorder=1)

        all_sizes = sorted({int(k) for m in models for k in data[m].get(bv, {})})

        for model_v in models:
            n_dict = data[model_v].get(bv, {})
            sizes  = sorted(int(k) for k in n_dict)
            if not sizes:
                continue
            means  = [n_dict[str(n)]["mean"] * 100 for n in sizes]
            stds   = [n_dict[str(n)]["std"]  * 100 for n in sizes]
            colour = COLOURS.get(model_v)
            label  = MODEL_LABELS.get(model_v, model_v)
            ax.plot(sizes, means, marker="o", linewidth=1.8, color=colour, label=label, zorder=3)
            ax.fill_between(
                sizes,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15, color=colour, zorder=2,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(all_sizes)
        ax.set_xticklabels([str(n) for n in all_sizes])
        ax.set_xlabel("N trials (log₂ scale)")
        ax.set_title(DATASET_TITLES.get(bv, bv))
        ax.set_ylim(40, 105)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out = pathlib.Path(args.dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
