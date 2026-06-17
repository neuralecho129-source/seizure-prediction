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

# ── 1. Get EDF file path from command line, or use default ────────────────────
edf_path = sys.argv[1] if len(sys.argv) > 1 else "data/chb01_03.edf"
print(f"Loading: {edf_path}")

# ── 2. Load + filter (same as label_windows.py) ───────────────────────────────
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
raw.filter(l_freq=1, h_freq=40, verbose=False)

sfreq = raw.info["sfreq"]
data = raw.get_data()  # shape: (n_channels, n_samples)

n_channels, n_samples = data.shape
window_samples = int(WINDOW_SECONDS * sfreq)
n_windows = n_samples // window_samples

print(f"Recording: {n_samples / sfreq / 60:.1f} minutes  |  {n_windows} windows of {WINDOW_SECONDS}s each")

# ── 3. Extract features from every window ─────────────────────────────────────
def extract_features(window):
    band_powers = {band: [] for band in BANDS}
    max_amplitudes = []

    for ch in range(window.shape[0]):
        signal = window[ch]
        freqs, psd = welch(signal, fs=sfreq, nperseg=256)

        total_power = sum(
            np.sum(psd[(freqs >= low) & (freqs <= high)])
            for low, high in BANDS.values()
        )

        for band, (low, high) in BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            relative = np.sum(psd[mask]) / total_power if total_power > 0 else 0
            band_powers[band].append(relative)

        max_amplitudes.append(np.max(np.abs(signal)))

    return [np.mean(band_powers[b]) for b in BANDS] + [np.mean(max_amplitudes)]

print("Extracting features...", end=" ", flush=True)
X = []
for i in range(n_windows):
    window = data[:, i * window_samples:(i + 1) * window_samples]
    X.append(extract_features(window))
X = np.array(X)
print("done.")

# ── 4. Load model and predict ─────────────────────────────────────────────────
model = joblib.load("seizure_model.pkl")
predictions = model.predict(X)

# ── 5. Print timeline ─────────────────────────────────────────────────────────
ICONS = {"normal": "  ", "pre-seizure": "! ", "seizure": "!!"}

print(f"\n{'Window':<8} {'Time':<18} {'Prediction'}")
print("-" * 40)

for i, pred in enumerate(predictions):
    start = int(i * WINDOW_SECONDS)
    end   = int(start + WINDOW_SECONDS)
    icon  = ICONS.get(pred, "  ")
    flag  = " <-- ALERT" if pred == "pre-seizure" else (" <-- SEIZURE" if pred == "seizure" else "")
    print(f"  {i+1:<6} {start:>4}s - {end:<6}s  {icon}{pred}{flag}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
unique, counts = np.unique(predictions, return_counts=True)
totals = dict(zip(unique, counts))

print("\n-- Summary --")
print(f"  normal:      {totals.get('normal', 0)} windows")
print(f"  pre-seizure: {totals.get('pre-seizure', 0)} windows")
print(f"  seizure:     {totals.get('seizure', 0)} windows")
