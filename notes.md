# Seizure Prediction Project — Notes

## The big idea
Building a wearable that uses EEG (brain signals) + other vitals to warn someone
~30 min before an epileptic seizure. Right now we're building and testing the
"brain" (the prediction algorithm) on a computer using real recorded EEG data,
before touching any hardware.

## Roadmap
1. **Understand the data** ✅ — load/plot EEG, see what a seizure looks like
2. **Clean and chop the data** ✅ — split recording into labeled time windows
3. **Extract features** ⬅️ next — turn each window into descriptive numbers
4. **Machine learning** — train + test a model that spots pre-seizure patterns
5. **Build the warning app** — wrap the model so it can send alerts

Beyond phase 5 (long-term, low-pressure): test on more patients/seizures,
research wearable EEG hardware (Muse, OpenBCI), live data pipeline, phone app,
real-world testing.

## Key vocab
- **EEG**: a recording of electrical activity (voltage) from the brain over time
- **Frequency bands**: delta, theta, alpha, beta, gamma — slowest to fastest brain waves
- **Amplitude / intensity**: how big the wiggles in the signal are
- **Synchronization**: multiple brain regions firing in the same rhythm at once
- **What a seizure looks like on EEG**: amplitude spikes way up, and the signal
  becomes more rhythmic/synchronized instead of random
- **Preictal**: the period before a seizure (we used 30 minutes)
- **Interictal**: normal, between-seizure brain activity
- **Feature**: a descriptive number calculated from raw data (e.g., "power in the alpha band")
- **Hyperparameter**: a setting we choose ourselves (e.g., window size = 10 seconds)

## Code concepts learned (MNE + numpy)
- `mne.io.read_raw_edf(path, preload=True)` — load an EEG recording
- `raw.copy().pick([...])` — grab specific channel(s)
- `raw.copy().crop(tmin=X, tmax=Y)` — keep only the time range between X and Y seconds
- `raw.copy().filter(l_freq=1, h_freq=40)` — clean signal, keep 1-40 Hz (brain activity range)
- `raw.get_data()` — get the actual voltage numbers as a numpy array (tiny decimals = microvolts)
- `np.mean()`, `np.max(np.abs(...))` — basic stats to compare normal vs. seizure signals

## Project files
- `data/chb01_03.edf` — real 1-hour EEG recording (CHB-MIT database via PhysioNet),
  contains one seizure from 2996s-3036s
- `data/chb01-summary.txt` — "answer key" with seizure timing
- `plot_eeg.py` — loads data, plots normal vs. seizure moments, saves as PNGs
- `normal_moment.png` / `seizure_moment.png` — visual comparison
- `explore_eeg.py` — hands-on exercises (pick channel, crop, get_data, filter, compare stats)
- `label_windows.py` — chops recording into 360 ten-second windows, labels each
  normal/pre-seizure/seizure, saves to `windows_labeled.npz`
- `windows_labeled.npz` — 360 labeled raw chunks (input for Phase 3)

## Phase 2 results (chb01_03.edf)
- 360 total windows (10 seconds each)
- 176 normal, 180 pre-seizure, 4 seizure

## Next session: Phase 3
Turn each of the 360 raw windows into a row of descriptive numbers (features) —
frequency band power (delta/theta/alpha/beta/gamma) + amplitude stats — to feed
into a machine learning model in Phase 4.
