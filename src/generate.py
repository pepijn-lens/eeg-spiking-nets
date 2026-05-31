"""
Synthetic ERD dataset generator — NiSNN-A BCIC IV 2a input interface.

Produces trials shaped (C, S, T) matching the NiSNN-A BCIC IV 2a setup:
  C=20 channels, S=20 timepieces, T=20 steps/timepiece → D=400 total samples.
Sampling rate after downsampling: 250 Hz × (400/750) ≈ 133.3 Hz
(NiSNN-A states 250 Hz original / 400 steps per 3 s trial; we adopt fs=250/750*400.)

Class identity is encoded *only* as mu/beta band-power modulation during a fixed
window on a contralateral channel subset — an ERD analogue.  The base signal
carries no class information.
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.signal import butter, sosfiltfilt


# ── helpers ──────────────────────────────────────────────────────────────────

def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via inverse-FFT shaping of white noise."""
    white = rng.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0  # avoid divide-by-zero at DC
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    spectrum[0] = 0.0  # zero DC
    return np.fft.irfft(spectrum, n=n_samples)


def _bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def _band_power(signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> float:
    """RMS power in a frequency band for a 1-D signal."""
    sos = _bandpass_sos(low_hz, high_hz, fs)
    filtered = sosfiltfilt(sos, signal)
    return float(np.sqrt(np.mean(filtered ** 2)))


# ── channel layout ───────────────────────────────────────────────────────────
# BCIC IV 2a channels: FC-3/1/z/2/4, C-5/3/1/z/2/4/6, CP-3/1/z/2/4, P-1/z/2
# (20 total, matching NiSNN-A Table I channel count)
CHANNEL_NAMES = [
    "FC3", "FC1", "FCz", "FC2", "FC4",           # 0-4   frontocentral
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4", "C6",  # 5-11  central
    "CP3", "CP1", "CPz", "CP2", "CP4",           # 12-16 centroparietal
    "P1",  "Pz",  "P2",                           # 17-19 parietal
]
assert len(CHANNEL_NAMES) == 20

# Contralateral ERD subsets (hand MI physiology: left-hand → right hemisphere, right-hand → left)
# Left-hand class A modulates right-hemisphere channels (C4, CP4, FC4, C6, CP2, FC2)
# Right-hand class B modulates left-hemisphere channels  (C3, CP3, FC3, C5, CP1, FC1)
LEFT_MODULATED_CHANNELS  = [CHANNEL_NAMES.index(c) for c in ["FC4","FC2","C6","C4","C2","CP4","CP2"]]
RIGHT_MODULATED_CHANNELS = [CHANNEL_NAMES.index(c) for c in ["FC3","FC1","C5","C3","C1","CP3","CP1"]]


# ── main generator ────────────────────────────────────────────────────────────

def generate_dataset(
    n_trials: int = 288,          # match BCIC IV 2a per-subject count
    n_channels: int = 20,
    fs: float = 400 / 3.0,        # effective rate: 400 steps over 3 s ≈ 133.3 Hz
    S: int = 20,                   # timepieces
    T: int = 20,                   # steps per timepiece
    mu_band: tuple = (8.0, 13.0),
    beta_band: tuple = (14.0, 30.0),
    mod_window: tuple = (0.5, 2.5),  # seconds into trial where ERD is active
    modulation_depth: float = 0.7,   # fraction of band power suppressed (0=none, 1=full)
    noise_level: float = 1.0,        # amplitude scale for base noise
    seed: int = 42,
) -> dict:
    """
    Returns a dict with keys:
      trials  : np.ndarray  shape (n_trials, C, S, T)
      labels  : np.ndarray  shape (n_trials,)  0=left, 1=right
      config  : dict
    """
    rng = np.random.default_rng(seed)
    D = S * T  # total samples per trial

    # pre-compute sample indices for modulation window
    t_axis = np.arange(D) / fs
    mod_mask = (t_axis >= mod_window[0]) & (t_axis <= mod_window[1])

    # build bandpass filters once
    mu_sos   = _bandpass_sos(*mu_band,   fs)
    beta_sos = _bandpass_sos(*beta_band, fs)

    trials = np.empty((n_trials, n_channels, S, T), dtype=np.float32)
    labels = np.empty(n_trials, dtype=np.int64)

    for i in range(n_trials):
        label = i % 2  # balanced classes
        labels[i] = label
        mod_channels = LEFT_MODULATED_CHANNELS if label == 0 else RIGHT_MODULATED_CHANNELS

        for c in range(n_channels):
            base = noise_level * _pink_noise(D, rng)

            if c in mod_channels:
                # isolate mu and beta components of base signal
                mu_component   = sosfiltfilt(mu_sos,   base)
                beta_component = sosfiltfilt(beta_sos, base)

                # suppress both bands in the modulation window
                suppression = np.ones(D)
                suppression[mod_mask] = 1.0 - modulation_depth

                base = (
                    base
                    - mu_component   * (1 - suppression)
                    - beta_component * (1 - suppression)
                )

            # reshape flat D → (S, T) and store
            trials[i, c] = base.reshape(S, T)

    config = dict(
        n_trials=n_trials,
        n_channels=n_channels,
        fs=fs,
        S=S,
        T=T,
        mu_band=list(mu_band),
        beta_band=list(beta_band),
        mod_window=list(mod_window),
        modulation_depth=modulation_depth,
        noise_level=noise_level,
        seed=seed,
        channel_names=CHANNEL_NAMES,
        left_modulated=LEFT_MODULATED_CHANNELS,
        right_modulated=RIGHT_MODULATED_CHANNELS,
    )

    return dict(trials=trials, labels=labels, config=config)


def save_dataset(data: dict, out_dir: pathlib.Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}_trials.npy",  data["trials"])
    np.save(out_dir / f"{name}_labels.npy",  data["labels"])
    with open(out_dir / f"{name}_config.json", "w") as f:
        json.dump(data["config"], f, indent=2)
    print(f"Saved {name}: trials={data['trials'].shape}, labels={data['labels'].shape} → {out_dir}")


def sanity_check(data: dict) -> None:
    """Verify band power genuinely differs between classes in modulated channels."""
    trials = data["trials"]
    labels = data["labels"]
    cfg    = data["config"]
    fs     = cfg["fs"]
    mu     = cfg["mu_band"]
    beta   = cfg["beta_band"]

    left_mask  = labels == 0
    right_mask = labels == 1

    # pick one modulated channel from each class's suppressed set
    ch_A = cfg["left_modulated"][0]   # suppressed when label=0
    ch_B = cfg["right_modulated"][0]  # suppressed when label=1

    def mean_bp(ch, mask, band):
        flat = trials[mask, ch].reshape(mask.sum(), -1)
        return np.mean([_band_power(row, fs, *band) for row in flat])

    print("\n── Sanity check: band power by class ──")
    print(f"Channel {CHANNEL_NAMES[ch_A]} (suppressed in class LEFT=0):")
    print(f"  mu  power  class 0 (should be LOW):  {mean_bp(ch_A, left_mask,  mu):.4f}")
    print(f"  mu  power  class 1 (should be HIGH): {mean_bp(ch_A, right_mask, mu):.4f}")
    print(f"  beta power class 0 (should be LOW):  {mean_bp(ch_A, left_mask,  beta):.4f}")
    print(f"  beta power class 1 (should be HIGH): {mean_bp(ch_A, right_mask, beta):.4f}")
    print(f"\nChannel {CHANNEL_NAMES[ch_B]} (suppressed in class RIGHT=1):")
    print(f"  mu  power  class 0 (should be HIGH): {mean_bp(ch_B, left_mask,  mu):.4f}")
    print(f"  mu  power  class 1 (should be LOW):  {mean_bp(ch_B, right_mask, mu):.4f}")
    print(f"  beta power class 0 (should be HIGH): {mean_bp(ch_B, left_mask,  beta):.4f}")
    print(f"  beta power class 1 (should be LOW):  {mean_bp(ch_B, right_mask, beta):.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ERD dataset")
    parser.add_argument("--variant",          default="control", choices=["control", "stress"])
    parser.add_argument("--n_trials",         type=int,   default=288)
    parser.add_argument("--modulation_depth", type=float, default=None,
                        help="Override depth (default: 0.7 for control, 0.15 for stress)")
    parser.add_argument("--noise_level",      type=float, default=None,
                        help="Override noise amplitude (default: 1.0 for control, 3.0 for stress)")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--out_dir",          type=str,   default=None)
    args = parser.parse_args()

    defaults = {
        "control": dict(modulation_depth=0.7,  noise_level=1.0),
        "stress":  dict(modulation_depth=0.15, noise_level=3.0),
    }
    depth = args.modulation_depth if args.modulation_depth is not None else defaults[args.variant]["modulation_depth"]
    noise = args.noise_level      if args.noise_level      is not None else defaults[args.variant]["noise_level"]
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path("data") / args.variant

    data = generate_dataset(
        n_trials=args.n_trials,
        modulation_depth=depth,
        noise_level=noise,
        seed=args.seed,
    )
    sanity_check(data)
    save_dataset(data, out_dir, args.variant)


if __name__ == "__main__":
    main()
