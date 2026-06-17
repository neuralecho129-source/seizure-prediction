import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ── 1. Load features ──────────────────────────────────────────────────────────
df = pd.read_csv("features.csv")

FEATURE_COLS = ["delta_power", "theta_power", "alpha_power",
                "beta_power", "gamma_power", "max_amplitude"]
TEST_FILE    = "chb01_04"   # held out — model never trains on this

train_df = df[df["source"] != TEST_FILE]
test_df  = df[df["source"] == TEST_FILE]

X_train, y_train = train_df[FEATURE_COLS].values, train_df["label"].values
X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df["label"].values

print(f"Training on {len(X_train)} windows from {train_df['source'].nunique()} recordings")
print(f"Testing  on {len(X_test)}  windows from {TEST_FILE}")
print()
for label, count in zip(*np.unique(y_train, return_counts=True)):
    print(f"  train {label}: {count}")

# ── 2. Train ──────────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
print("\nModel trained.")

# ── 3. Evaluate on held-out recording ─────────────────────────────────────────
y_pred      = model.predict(X_test)
overall_acc = (y_pred == y_test).mean() * 100
print(f"\nAccuracy on {TEST_FILE}: {overall_acc:.1f}%")

print("\nPer-class results:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix (rows = true, cols = predicted):")
labels    = sorted(set(y_test))
cm        = confusion_matrix(y_test, y_pred, labels=labels)
col_width = 14
print(" " * col_width + "".join(f"{l:>{col_width}}" for l in labels))
for row_label, row in zip(labels, cm):
    print(f"{row_label:<{col_width}}" + "".join(f"{v:>{col_width}}" for v in row))

# ── 4. Feature importances ────────────────────────────────────────────────────
print("\nFeature importances:")
for name, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
    bar = "#" * int(imp * 40)
    print(f"  {name:<18} {imp:.3f}  {bar}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
joblib.dump(model, "seizure_model.pkl")
print("\nModel saved to seizure_model.pkl")
