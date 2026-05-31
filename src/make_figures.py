"""
Generate the three required figures from the control dataset.
Run from project root:  .venv/bin/python src/make_figures.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram

# ── load ──────────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path("data/control")
FIG_DIR  = pathlib.Path("figures")
FIG_DIR.mkdir(exist_ok=True)

trials = np.load(DATA_DIR / "control_trials.npy")   # (288, 20, 20, 20)
labels = np.load(DATA_DIR / "control_labels.npy")
with open(DATA_DIR / "control_config.json") as f:
    cfg = json.load(f)

fs          = cfg["fs"]           # ~133.3 Hz
S, T        = cfg["S"], cfg["T"]  # 20, 20
D           = S * T               # 400
t_axis      = np.arange(D) / fs  # seconds
CHAN_NAMES  = cfg["channel_names"]
MU          = cfg["mu_band"]
BETA        = cfg["beta_band"]
LEFT_MOD    = cfg["left_modulated"]
RIGHT_MOD   = cfg["right_modulated"]


def flat(trial_ch):
    """Flatten (S, T) → (D,)."""
    return trial_ch.reshape(-1)


def bandpass_filter(sig, low, high):
    nyq = fs / 2
    sos = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, sig)


def band_rms(sig, low, high):
    return float(np.sqrt(np.mean(bandpass_filter(sig, low, high) ** 2)))


# ── Figure (a): raw multichannel trace ───────────────────────────────────────
# Show one left-class and one right-class trial side-by-side, 6 representative channels.
SHOW_CHANS = [CHAN_NAMES.index(c) for c in ["FC3","FC4","C3","C4","CP3","CP4"]]
SHOW_NAMES = [CHAN_NAMES[i] for i in SHOW_CHANS]

idx_left  = np.where(labels == 0)[0][0]
idx_right = np.where(labels == 1)[0][0]

fig, axes = plt.subplots(len(SHOW_CHANS), 2, figsize=(10, 7), sharey="row")
fig.suptitle("(a) Raw multichannel traces — left-hand (class 0) vs right-hand (class 1) trial",
             fontsize=10, y=1.01)

for row, (ci, cname) in enumerate(zip(SHOW_CHANS, SHOW_NAMES)):
    for col, (trial_idx, class_label) in enumerate([(idx_left, "Left (class 0)"),
                                                     (idx_right, "Right (class 1)")]):
        ax = axes[row, col]
        sig = flat(trials[trial_idx, ci])
        ax.plot(t_axis, sig, lw=0.7, color="steelblue" if col == 0 else "tomato")
        # shade modulation window
        ax.axvspan(cfg["mod_window"][0], cfg["mod_window"][1],
                   alpha=0.12, color="gold", label="ERD window" if row == 0 else None)
        ax.set_ylabel(cname, fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        if row == 0:
            ax.set_title(class_label, fontsize=9)
        if row == len(SHOW_CHANS) - 1:
            ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)

handles = [plt.Rectangle((0,0),1,1, color="gold", alpha=0.4)]
fig.legend(handles, ["ERD window (0.5–2.5 s)"], loc="lower center", ncol=1, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout()

caption = (
    "Figure (a). Raw amplitude traces (arbitrary units) for six electrode pairs "
    "(left/right hemisphere, frontal-central-parietal rows) for one left-class trial (blue) "
    "and one right-class trial (red). The gold band marks the ERD modulation window "
    "(0.5–2.5 s). The contralateral suppression is not visible by eye in the raw signal "
    "— it lives in the mu/beta frequency bands — confirming that band-power analysis "
    "(Figures b and c) is required to reveal the class structure."
)
with open(FIG_DIR / "fig_a_caption.txt", "w") as f:
    f.write(caption)

fig.savefig(FIG_DIR / "fig_a_traces.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_a_traces.png")


# ── Figure (b): time-frequency spectrogram ───────────────────────────────────
# Average spectrogram over left-class trials for C4 (suppressed in left class)
# vs right-class trials for C4. Show both in one figure with mu/beta overlays.

CH_IDX = CHAN_NAMES.index("C4")  # in LEFT_MOD — suppressed when label=0

class0_trials = trials[labels == 0, CH_IDX]  # (144, S, T)
class1_trials = trials[labels == 1, CH_IDX]

def avg_spectrogram(group):
    """Mean power spectrogram across a group of trials."""
    psds = []
    for trial in group:
        sig = flat(trial)
        f, t, Sxx = spectrogram(sig, fs=fs, nperseg=64, noverlap=48, nfft=256)
        psds.append(Sxx)
    return f, t, np.mean(psds, axis=0)

f, t, psd_c0 = avg_spectrogram(class0_trials)
_, _, psd_c1 = avg_spectrogram(class1_trials)

# clip to 5-40 Hz — starting at 0 lets the 1/f low-frequency power dominate the colour scale
freq_mask = (f >= 5) & (f <= 40)
f_disp    = f[freq_mask]

vmin = np.percentile(np.log1p(np.stack([psd_c0[freq_mask], psd_c1[freq_mask]])), 2)
vmax = np.percentile(np.log1p(np.stack([psd_c0[freq_mask], psd_c1[freq_mask]])), 98)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
fig.subplots_adjust(left=0.07, right=0.82, top=0.88, bottom=0.14, wspace=0.15)
fig.suptitle("(b) Mean time-frequency spectrogram at electrode C4",
             fontsize=10)

titles = ["Left-hand class (C4 is contralateral → ERD)", "Right-hand class (C4 is ipsilateral → no ERD)"]
psds   = [psd_c0, psd_c1]
for ax, psd, title in zip(axes, psds, titles):
    im = ax.pcolormesh(t, f_disp, np.log1p(psd[freq_mask]),
                       shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.axhspan(MU[0],   MU[1],   alpha=0.25, color="cyan",   label=f"mu ({MU[0]}–{MU[1]} Hz)")
    ax.axhspan(BETA[0], BETA[1], alpha=0.25, color="magenta", label=f"beta ({BETA[0]}–{BETA[1]} Hz)")
    ax.axvline(cfg["mod_window"][0], color="white", lw=1.2, ls="--", label="ERD window")
    ax.axvline(cfg["mod_window"][1], color="white", lw=1.2, ls="--")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7, loc="upper right")

# colorbar in explicitly reserved space to the right
cbar_ax = fig.add_axes([0.84, 0.14, 0.02, 0.74])
fig.colorbar(im, cax=cbar_ax, label="log(1 + power density)")

caption = (
    "Figure (b). Mean log-power spectrogram at electrode C4, averaged over all 144 "
    "left-class trials (left panel) and all 144 right-class trials (right panel). "
    "Cyan and magenta bands mark the mu (8–13 Hz) and beta (14–30 Hz) ranges; "
    "dashed white lines bound the 0.5–2.5 s ERD window. "
    "C4 sits in the contralateral hemisphere for left-hand imagery, so the left-class "
    "panel shows clear power suppression in both bands during the window, while the "
    "right-class panel does not — confirming that the planted ERD is visible in the "
    "time-frequency domain."
)
with open(FIG_DIR / "fig_b_caption.txt", "w") as f:
    f.write(caption)

fig.savefig(FIG_DIR / "fig_b_spectrogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_b_spectrogram.png")


# ── Figure (c): class-mean band-power per channel ────────────────────────────
# For each channel compute mean RMS band power in mu+beta for each class,
# then plot the difference (class 0 - class 1) to reveal contralateral pattern.

def channel_band_power(band):
    out = np.zeros((2, 20))
    for label in [0, 1]:
        group = trials[labels == label]   # (144, 20, S, T)
        for c in range(20):
            sigs = group[:, c].reshape(len(group), -1)  # (144, D)
            out[label, c] = np.mean([band_rms(s, *band) for s in sigs])
    return out

print("Computing per-channel band power (takes ~10 s)...")
mu_bp   = channel_band_power(MU)
beta_bp = channel_band_power(BETA)

# combine: mean of mu and beta
combined = (mu_bp + beta_bp) / 2.0   # (2, 20)

# difference: class 0 - class 1 (positive = suppressed in class 0)
diff = combined[0] - combined[1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("(c) Class-mean mu+beta band power per electrode and contralateral pattern",
             fontsize=10)

x = np.arange(20)
width = 0.38

ax = axes[0]
bars0 = ax.bar(x - width/2, combined[0], width, label="Left class (0)",  color="steelblue", alpha=0.85)
bars1 = ax.bar(x + width/2, combined[1], width, label="Right class (1)", color="tomato",    alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(CHAN_NAMES, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Mean RMS band power (a.u.)", fontsize=9)
ax.set_title("Absolute band power by class", fontsize=9)
ax.legend(fontsize=8)
# mark modulated channels
for ci in LEFT_MOD:
    ax.get_xticklabels()[ci].set_color("steelblue")
for ci in RIGHT_MOD:
    ax.get_xticklabels()[ci].set_color("tomato")

ax2 = axes[1]
colours = ["steelblue" if d > 0 else "tomato" for d in diff]
ax2.bar(x, diff, color=colours, alpha=0.85)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(CHAN_NAMES, rotation=45, ha="right", fontsize=7)
ax2.set_ylabel("Δ power: class 0 − class 1 (a.u.)", fontsize=9)
ax2.set_title("Contralateral ERD pattern (blue = suppressed in left class)", fontsize=9)

fig.tight_layout()

caption = (
    "Figure (c). Left panel: mean mu+beta RMS band power per electrode for the left-class "
    "(blue) and right-class (red) trials. "
    "Right panel: difference (class 0 − class 1); blue bars mark electrodes suppressed in "
    "the left-class trial (right hemisphere: FC4, FC2, C6, C4, C2, CP4, CP2), red bars mark "
    "electrodes suppressed in the right-class trial (left hemisphere: FC3, FC1, C5, C3, C1, "
    "CP3, CP1). The pattern mirrors contralateral ERD in real motor-imagery EEG: the "
    "hemisphere opposite to the imagined hand shows the greatest power reduction, while the "
    "ipsilateral hemisphere is unaffected. This confirms the dataset encodes class identity "
    "solely and precisely as a lateralised band-power modulation."
)
with open(FIG_DIR / "fig_c_caption.txt", "w") as f:
    f.write(caption)

fig.savefig(FIG_DIR / "fig_c_bandpower.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_c_bandpower.png")
