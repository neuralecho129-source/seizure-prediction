import mne
import numpy as np

# ------------------------------------------------------------
# Step 1: Load and clean the recording (same as Phase 1)
# ------------------------------------------------------------
raw = mne.io.read_raw_edf("data/chb01_03.edf", preload=True)
raw.filter(l_freq=1, h_freq=40)

sfreq = raw.info["sfreq"]  # samples per second (256 Hz)
data = raw.get_data()      # shape: (23 channels, 921600 samples)

# ------------------------------------------------------------
# Step 2: Seizure timing info (from chb01-summary.txt)
# ------------------------------------------------------------
seizure_start = 2996  # seconds
seizure_end = 3036    # seconds
preictal_duration = 30 * 60  # 30 minutes, in seconds

# ------------------------------------------------------------
# Step 3: Slice the recording into 10-second windows
# ------------------------------------------------------------
window_seconds = 10
window_samples = int(window_seconds * sfreq)  # 2560 samples per window

n_channels, n_samples = data.shape
n_windows = n_samples // window_samples

windows = []
labels = []

for i in range(n_windows):
    start_sample = i * window_samples
    end_sample = start_sample + window_samples

    window_data = data[:, start_sample:end_sample]

    start_time = start_sample / sfreq
    end_time = end_sample / sfreq
    center_time = (start_time + end_time) / 2

    if seizure_start <= center_time <= seizure_end:
        label = "seizure"
    elif (seizure_start - preictal_duration) <= center_time < seizure_start:
        label = "pre-seizure"
    else:
        label = "normal"

    windows.append(window_data)
    labels.append(label)

windows = np.array(windows)  # shape: (360, 23, 2560)
labels = np.array(labels)

# ------------------------------------------------------------
# Step 4: Print a summary so we can sanity-check it
# ------------------------------------------------------------
print("Total windows:", len(labels))
print("  normal:     ", np.sum(labels == "normal"))
print("  pre-seizure:", np.sum(labels == "pre-seizure"))
print("  seizure:    ", np.sum(labels == "seizure"))
print("\nShape of 'windows' array:", windows.shape)
print("(meaning: 360 windows, each with 23 channels, each 2560 samples long)")

# ------------------------------------------------------------
# Step 5: Save everything to one file for Phase 3
# ------------------------------------------------------------
np.savez_compressed("windows_labeled.npz", windows=windows, labels=labels)
print("\nSaved windows_labeled.npz")
