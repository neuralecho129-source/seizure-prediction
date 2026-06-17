import mne
import matplotlib.pyplot as plt

# Load the recording (MNE knows how to read .edf brain-wave files)
raw = mne.io.read_raw_edf("data/chb01_03.edf", preload=True)

print("\nChannels (sensors on the head):", raw.ch_names)
print("Recording length (seconds):", raw.times[-1])
print("Sampling rate (data points per second):", raw.info["sfreq"])

# This recording has ONE seizure, from the summary file:
#   Seizure start: 2996 seconds into the recording
#   Seizure end:   3036 seconds into the recording
seizure_start = 2996
seizure_end = 3036

# --- Plot 1: a calm, normal moment (5 minutes before the seizure) ---
normal_window_start = seizure_start - 300
fig1 = raw.copy().crop(tmin=normal_window_start, tmax=normal_window_start + 20).plot(
    show=False, title="NORMAL brain activity (calm moment)"
)
fig1.savefig("normal_moment.png", dpi=120)
print("Saved normal_moment.png")

# --- Plot 2: the actual seizure moment ---
fig2 = raw.copy().crop(tmin=seizure_start, tmax=seizure_start + 20).plot(
    show=False, title="SEIZURE happening right now!"
)
fig2.savefig("seizure_moment.png", dpi=120)
print("Saved seizure_moment.png")
