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

## 4. Override parameters (optional)

```bash
# weaker control variant
.venv/bin/python src/generate.py --variant control \
    --modulation_depth 0.4 --noise_level 1.5 --seed 0 \
    --out_dir data/custom

# see all options
.venv/bin/python src/generate.py --help
```
