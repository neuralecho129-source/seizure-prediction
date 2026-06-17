import sys
import numpy as np
import mne
import joblib
from scipy.signal import welch

# ── Settings (must match extract_features.py) ─────────────────────────────────
WINDOW_SECONDS = 10
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}

# ── Recording info (from chb01-summary.txt) ───────────────────────────────────
EDF_PATH       = "data/chb01_15.edf"
SEIZURE_START  = 1732   # seconds
SEIZURE_END    = 1772   # seconds
PREICTAL_SECS  = 30 * 60  # 30 minutes

# ── 1. Load + filter ──────────────────────────────────────────────────────────
print(f"Loading {EDF_PATH} ...")
raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose=False)
raw.filter(l_freq=1, h_freq=40, verbose=False)

sfreq        = raw.info["sfreq"]
data         = raw.get_data()
window_samples = int(WINDOW_SECONDS * sfreq)
n_windows    = data.shape[1] // window_samples
print(f"Recording: {data.shape[1] / sfreq / 60:.1f} min  |  {n_windows} windows")

# ── 2. Extract features ───────────────────────────────────────────────────────
def extract_features(window):
    band_powers   = {band: [] for band in BANDS}
    max_amplitudes = []
    for ch in range(window.shape[0]):
        signal = window[ch]
        freqs, psd = welch(signal, fs=sfreq, nperseg=256)
        total = sum(
            np.sum(psd[(freqs >= lo) & (freqs <= hi)])
            for lo, hi in BANDS.values()
        )
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            band_powers[band].append(np.sum(psd[mask]) / total if total > 0 else 0)
        max_amplitudes.append(np.max(np.abs(signal)))
    return [np.mean(band_powers[b]) for b in BANDS] + [np.mean(max_amplitudes)]

print("Extracting features...", end=" ", flush=True)
X = []
for i in range(n_windows):
    window = data[:, i * window_samples:(i + 1) * window_samples]
    X.append(extract_features(window))
X = np.array(X)
print("done.")

# ── 3. Build ground truth labels ──────────────────────────────────────────────
true_labels = []
for i in range(n_windows):
    center = (i * WINDOW_SECONDS) + (WINDOW_SECONDS / 2)
    if SEIZURE_START <= center <= SEIZURE_END:
        true_labels.append("seizure")
    elif (SEIZURE_START - PREICTAL_SECS) <= center < SEIZURE_START:
        true_labels.append("pre-seizure")
    else:
        true_labels.append("normal")
true_labels = np.array(true_labels)

# ── 4. Predict ────────────────────────────────────────────────────────────────
model      = joblib.load("seizure_model.pkl")
predictions = model.predict(X)

# ── 5. Accuracy per class ─────────────────────────────────────────────────────
overall_acc = (predictions == true_labels).mean() * 100
print(f"\nOverall accuracy: {overall_acc:.1f}%")
print()

for label in ["normal", "pre-seizure", "seizure"]:
    mask       = true_labels == label
    total      = mask.sum()
    if total == 0:
        continue
    correct    = (predictions[mask] == true_labels[mask]).sum()
    acc        = correct / total * 100
    print(f"  {label:<14} {correct}/{total} correct  ({acc:.0f}%)")

# ── 6. Did it catch the pre-seizure period? ───────────────────────────────────
pre_mask      = true_labels == "pre-seizure"
pre_predicted = predictions[pre_mask]
caught        = (pre_predicted == "pre-seizure").sum()
total_pre     = pre_mask.sum()

print(f"\nPre-seizure windows caught: {caught}/{total_pre}")
print(f"False alarms (normal predicted as pre-seizure): "
      f"{((predictions == 'pre-seizure') & (true_labels == 'normal')).sum()}")
