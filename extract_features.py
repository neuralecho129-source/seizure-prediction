import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

# ── All recordings with their seizure timing (from chb01-summary.txt) ─────────
RECORDINGS = [
    {"file": "data/chb01_03.edf", "seizure_start": 2996, "seizure_end": 3036},
    {"file": "data/chb01_04.edf", "seizure_start": 1467, "seizure_end": 1494},
    {"file": "data/chb01_15.edf", "seizure_start": 1732, "seizure_end": 1772},
    {"file": "data/chb01_16.edf", "seizure_start": 1015, "seizure_end": 1066},
    {"file": "data/chb01_18.edf", "seizure_start": 1720, "seizure_end": 1810},
    {"file": "data/chb01_21.edf", "seizure_start":  327, "seizure_end":  420},
    {"file": "data/chb01_26.edf", "seizure_start": 1862, "seizure_end": 1963},
]

WINDOW_SECONDS  = 10
PREICTAL_SECS   = 30 * 60  # 30 minutes
BANDS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}

def extract_features(window, sfreq):
    band_powers    = {band: [] for band in BANDS}
    max_amplitudes = []

    for ch in range(window.shape[0]):
        signal     = window[ch]
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

# ── Process every recording ────────────────────────────────────────────────────
all_rows = []

for rec in RECORDINGS:
    path          = rec["file"]
    seizure_start = rec["seizure_start"]
    seizure_end   = rec["seizure_end"]
    source        = path.split("/")[-1].replace(".edf", "")

    print(f"Processing {source} ...", end=" ", flush=True)

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.filter(l_freq=1, h_freq=40, verbose=False)

    sfreq          = raw.info["sfreq"]
    data           = raw.get_data()
    window_samples = int(WINDOW_SECONDS * sfreq)
    n_windows      = data.shape[1] // window_samples

    # pre-seizure window: cap at recording start so short files aren't mislabeled
    preictal_start = max(0, seizure_start - PREICTAL_SECS)

    for i in range(n_windows):
        center = i * WINDOW_SECONDS + WINDOW_SECONDS / 2

        if seizure_start <= center <= seizure_end:
            label = "seizure"
        elif preictal_start <= center < seizure_start:
            label = "pre-seizure"
        else:
            label = "normal"

        window   = data[:, i * window_samples:(i + 1) * window_samples]
        features = extract_features(window, sfreq)

        row = {f: v for f, v in zip(
            ["delta_power", "theta_power", "alpha_power",
             "beta_power", "gamma_power", "max_amplitude"],
            features
        )}
        row["label"]  = label
        row["source"] = source
        all_rows.append(row)

    counts = {l: sum(1 for r in all_rows if r["source"] == source and r["label"] == l)
              for l in ["normal", "pre-seizure", "seizure"]}
    print(f"normal={counts['normal']}  pre-seizure={counts['pre-seizure']}  seizure={counts['seizure']}")

# ── Save ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
df.to_csv("features.csv", index=False)

print(f"\nTotal windows: {len(df)}")
print(df.groupby("label")[["delta_power","theta_power","alpha_power","beta_power","gamma_power"]].mean().round(3))
print("\nSaved features.csv")
