# Seizure Prediction from EEG

A machine learning pipeline that predicts seizures from EEG (brain signal) recordings,
classifying each 10-second window as **normal**, **pre-seizure**, or **seizure** based on
frequency band features.

Built using real clinical EEG data from the CHB-MIT Scalp EEG Database (PhysioNet).

---

## Results

| Recording | Accuracy | Pre-seizure caught | Seizure caught |
|-----------|----------|--------------------|----------------|
| chb01_15  | 98.1%    | 173/173 (100%)     | 4/4 (100%)     |
| chb01_04  | 47.8%    | 42/147 (29%)       | 2/2 (100%)     |

chb01_04 is a harder case — the seizure occurs only 24 minutes into the recording,
leaving no clean normal baseline before the pre-seizure window.

---

## How it works

1. **Load EEG** — reads `.edf` recording files using MNE, filters to 1–40 Hz
2. **Window** — chops the recording into 10-second windows
3. **Label** — marks each window normal / pre-seizure (30 min before) / seizure
4. **Extract features** — computes relative power in 5 frequency bands (delta, theta,
   alpha, beta, gamma) + max amplitude per window
5. **Train** — Random Forest classifier trained on 6 recordings from one patient
6. **Predict** — outputs a timestamped prediction for every window in a new recording

---

## Project structure

```
seizure_project/
├── data/
│   └── chb01-summary.txt     # seizure timing for each recording
├── label_windows.py          # Phase 2: chop recording into labeled windows
├── extract_features.py       # Phase 3: compute frequency features from windows
├── train_model.py            # Phase 4: train Random Forest, evaluate on held-out file
├── predict.py                # Phase 5: end-to-end prediction on a new EDF file
├── evaluate.py               # accuracy evaluation against known ground truth
├── features.csv              # extracted features for all 7 recordings
├── seizure_model.pkl         # trained model
├── normal_moment.png         # EEG plot — normal brain activity
├── seizure_moment.png        # EEG plot — seizure brain activity
└── notes.md                  # project log and vocab reference
```

---

## Setup

```bash
pip install mne numpy scipy pandas scikit-learn joblib
```

EEG data files (.edf) must be downloaded separately from
[PhysioNet CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
and placed in the `data/` folder.

---

## Usage

**Train the model:**
```bash
python extract_features.py   # regenerate features.csv from all recordings
python train_model.py        # train and evaluate
```

**Run prediction on a new recording:**
```bash
python predict.py data/chb01_15.edf
```

---

## Dataset

Shoeb, A. H. (2009). *Application of Machine Learning to Epileptic Seizure Onset
Detection and Treatment*. MIT PhD Thesis.

Available at: https://physionet.org/content/chbmit/1.0.0/
