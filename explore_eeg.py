import mne
import numpy as np

raw = mne.io.read_raw_edf("data/chb01_03.edf", preload=True)

seizure_start = 2996
seizure_end = 3036

# ============================================================
# EXERCISE 1: Look at ONE channel instead of all 23
# ============================================================
# raw.copy().pick(["FP1-F7"]) gives you just that one sensor.
one_channel = raw.copy().pick(["FP1-F7"])
print("Exercise 1: picked channel ->", one_channel.ch_names)


# ============================================================
# EXERCISE 2: Get the RAW NUMBERS (not a picture!)
# ============================================================
# raw.get_data() turns the signal into a numpy array of numbers.
# Shape = (number of channels, number of data points)
data, times = raw.get_data(return_times=True)
print("\nExercise 2:")
print("  data shape (channels x samples):", data.shape)
print("  first 5 numbers from channel 0:", data[0, :5])
print("  first 5 time stamps (seconds):", times[:5])


# ============================================================
# EXERCISE 3: Compare NORMAL vs SEIZURE using actual numbers
# ============================================================
# We'll grab channel 0's data during a normal moment and
# during the seizure, then compare them with simple stats.

normal_segment = raw.copy().crop(tmin=seizure_start - 300, tmax=seizure_start - 280)
seizure_segment = raw.copy().crop(tmin=seizure_start, tmax=seizure_start + 20)

normal_data = normal_segment.get_data()[5]   # channel 0 only
seizure_data = seizure_segment.get_data()[5]  # channel 0 only

print("\nExercise 3: comparing channel 0 (FP1-F7)")
print(f"  NORMAL  -> mean: {np.mean(normal_data):.6f}, max abs: {np.max(np.abs(normal_data)):.6f}")
print(f"  SEIZURE -> mean: {np.mean(seizure_data):.6f}, max abs: {np.max(np.abs(seizure_data)):.6f}")
print("  (max abs = how big the biggest wiggle is -> this is 'amplitude'/intensity)")


# ============================================================
# EXERCISE 4 (YOUR TURN): Try a filter to clean up the signal
# ============================================================
# TODO: Uncomment the line below and run the script.
# This removes very slow drift and very fast noise, keeping
# only the 1-40 Hz range (where most brain activity lives).
#
filtered = raw.copy().filter(l_freq=1, h_freq=40)
#print("Filtered first 5 numbers:", filtered.get_data()[0, :5])
# Then try printing filtered.get_data()[0, :5] and compare it
# to data[0, :5] from Exercise 2 -- the numbers should look
# a bit smoother / different.


# ============================================================
# EXERCISE 5 (YOUR TURN): Pick a DIFFERENT channel and time
# ============================================================
# TODO: Change "FP1-F7" to a different channel name from this list:
#   raw.ch_names
# and change seizure_start - 300 to a different number of
# seconds before the seizure. Re-run Exercise 3's comparison
# with your new channel/time and see if the numbers still
# show a clear difference between normal and seizure.
