# Reproducing the results

All commands run from the project root.

## 1. Environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Generate datasets

```bash
.venv/bin/python src/generate.py --variant control
.venv/bin/python src/generate.py --variant stress
```

Outputs written to `data/control/` and `data/stress/`:
- `*_trials.npy` — shape (288, 20, 20, 20), float32
- `*_labels.npy` — shape (288,), int64; 0 = left, 1 = right
- `*_config.json` — all generation parameters

The terminal prints a sanity check confirming band power is lower in the
modulated channels for the correct class.

## 3. Generate figures

```bash
.venv/bin/python src/make_figures.py
```

Outputs written to `figures/`:
- `fig_a_traces.png` — raw multichannel traces
- `fig_b_spectrogram.png` — time-frequency spectrogram at C4
- `fig_c_bandpower.png` — class-mean band power per channel

## 4. Run ablation experiments (c1–c4)

Install the ML dependencies (not needed for dataset generation or figures):

```bash
.venv/bin/pip install torch scikit-learn
```

Run all three model variants × two datasets (6 conditions, 5-fold CV each):

```bash
.venv/bin/python src/train.py
```

Outputs:
- Printed results table with mean ± std accuracy per condition
- `results.json` with per-fold accuracy for all conditions
- Hypothesis verdict printed automatically

Subset options:

```bash
# c1/c2 only — full model on control (sanity check + ceiling)
.venv/bin/python src/train.py --variants full --datasets control

# c3 ablation — encoder-only on both datasets
.venv/bin/python src/train.py --variants encoder_only

# c4 ablation — fixed encoder on both datasets
.venv/bin/python src/train.py --variants fixed_encoder

# longer run for better convergence
.venv/bin/python src/train.py --epochs 200
```

Expected results (control dataset, 100 epochs):
- `full` (c1/c2): ~85–95% — clear 70% ERD modulation is easy to decode
- `encoder_only` (c3): similar to full → supports the hypothesis
- `fixed_encoder` (c4): ~50% → learned encoder was doing the discriminative work

## 5. Override parameters (optional)

```bash
# weaker control variant
.venv/bin/python src/generate.py --variant control \
    --modulation_depth 0.4 --noise_level 1.5 --seed 0 \
    --out_dir data/custom

# see all options
.venv/bin/python src/generate.py --help
```
