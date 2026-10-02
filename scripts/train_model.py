"""
train_model.py
--------------
Trains a Decision Tree classifier (primary) and a Random Forest classifier
(comparison) on data/training_data.csv.

Evaluation
----------
- Stratified 80/20 train/test split
- Classification report: accuracy, precision, recall, F1 per fault class
- Feature importance table (top features)
- Decision Tree text representation (export_text) to confirm explainability

Output
------
- data/model.pkl      : serialised best model (Decision Tree or Random Forest)
- data/model_meta.json: feature column list and label mapping saved alongside
                        the model so the engine layer can load them consistently

Run from the project root:
    python scripts/train_model.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TRAINING_CSV = ROOT / "data" / "training_data.csv"
MODEL_PKL = ROOT / "data" / "model.pkl"
MODEL_META = ROOT / "data" / "model_meta.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_ACCURACY = 0.85

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(title: str = "") -> None:
    line = "-" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
sep("1. Loading dataset")

if not TRAINING_CSV.exists():
    print(f"ERROR: training data not found at {TRAINING_CSV}")
    sys.exit(1)

df = pd.read_csv(TRAINING_CSV)
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")

# Split features / target
TARGET_COL = "fault_id"
feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].to_numpy(dtype=np.float32)
y = df[TARGET_COL].to_numpy()

print(f"\n  Feature count : {len(feature_cols)}")
print(f"  Feature names : {feature_cols}")
print(f"\n  Class distribution:")
for label, count in sorted(zip(*np.unique(y, return_counts=True))):
    print(f"    {label}: {count} sample(s)")

# ---------------------------------------------------------------------------
# 2. Train / test split (stratified)
# ---------------------------------------------------------------------------
sep("2. Train / test split  (80 / 20, stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")

# ---------------------------------------------------------------------------
# 3. Train Decision Tree
# ---------------------------------------------------------------------------
sep("3. Decision Tree Classifier")

dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,       # allow the tree to grow to fit the training data
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

print(f"\n  Test accuracy : {acc_dt:.4f}  ({acc_dt*100:.1f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_dt, zero_division=0))

# 5-fold stratified cross-validation on the full dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores_dt = cross_val_score(dt, X, y, cv=cv, scoring="accuracy")
print(f"  5-fold CV accuracy: {cv_scores_dt.mean():.4f} +/- {cv_scores_dt.std():.4f}")
print(f"  CV scores per fold: {[round(s, 4) for s in cv_scores_dt]}")

# Feature importance
sep("  Decision Tree — Feature Importance")
fi_dt = sorted(zip(feature_cols, dt.feature_importances_), key=lambda x: x[1], reverse=True)
print(f"  {'Feature':<35} Importance")
print(f"  {'-'*35} ----------")
for feat, imp in fi_dt:
    bar = "#" * int(imp * 40)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# Decision path (export_text) -- confirm explainability
sep("  Decision Tree — Text Representation (first 40 lines)")
tree_text = export_text(dt, feature_names=feature_cols)
for i, line in enumerate(tree_text.splitlines()[:40]):
    print(f"  {line}")
if len(tree_text.splitlines()) > 40:
    remaining = len(tree_text.splitlines()) - 40
    print(f"  ... ({remaining} more lines)")

# ---------------------------------------------------------------------------
# 4. Train Random Forest (comparison)
# ---------------------------------------------------------------------------
sep("4. Random Forest Classifier (comparison)")

rf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"\n  Test accuracy : {acc_rf:.4f}  ({acc_rf*100:.1f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

cv_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"  5-fold CV accuracy: {cv_scores_rf.mean():.4f} +/- {cv_scores_rf.std():.4f}")
print(f"  CV scores per fold: {[round(s, 4) for s in cv_scores_rf]}")

# Feature importance (Random Forest)
sep("  Random Forest — Feature Importance")
fi_rf = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
print(f"  {'Feature':<35} Importance")
print(f"  {'-'*35} ----------")
for feat, imp in fi_rf:
    bar = "#" * int(imp * 40)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ---------------------------------------------------------------------------
# 5. Model selection
# ---------------------------------------------------------------------------
sep("5. Model Selection")

print(f"  Decision Tree test accuracy : {acc_dt:.4f}  ({acc_dt*100:.1f}%)")
print(f"  Random Forest test accuracy : {acc_rf:.4f}  ({acc_rf*100:.1f}%)")

# Decision Tree is preferred for explainability unless RF is meaningfully better.
# "Meaningfully better" = RF accuracy > DT accuracy by more than 2 percentage points.
use_dt = acc_dt >= acc_rf - 0.02

if use_dt:
    best_model = dt
    best_name = "Decision Tree"
    best_acc = acc_dt
    print(f"\n  => Saving Decision Tree (preferred for explainability).")
else:
    best_model = rf
    best_name = "Random Forest"
    best_acc = acc_rf
    print(f"\n  => Saving Random Forest (meaningfully more accurate than Decision Tree).")

# Accuracy gate
if best_acc < MIN_ACCURACY:
    print(
        f"\n  WARNING: best model accuracy {best_acc*100:.1f}% is below the "
        f"{MIN_ACCURACY*100:.0f}% target. Consider augmenting training_data.csv."
    )
else:
    print(f"  Accuracy target ({MIN_ACCURACY*100:.0f}%) MET: {best_acc*100:.1f}%")

# ---------------------------------------------------------------------------
# 6. Save model and metadata
# ---------------------------------------------------------------------------
sep("6. Saving model and metadata")

joblib.dump(best_model, MODEL_PKL)
print(f"  Saved model  : {MODEL_PKL}")

# Build label list (sorted for determinism)
labels = sorted(set(y))

meta = {
    "model_type": best_name,
    "feature_columns": feature_cols,
    "target_column": TARGET_COL,
    "labels": labels,
    "test_accuracy": round(float(best_acc), 6),
    "cv_mean_accuracy": round(float(cv_scores_dt.mean() if use_dt else cv_scores_rf.mean()), 6),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "sklearn_version": __import__("sklearn").__version__,
}

with open(MODEL_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"  Saved metadata: {MODEL_META}")

# ---------------------------------------------------------------------------
# 7. Smoke test — reload model and run one prediction per class
# ---------------------------------------------------------------------------
sep("7. Smoke test -- reload model and predict one sample per fault class")

loaded_model = joblib.load(MODEL_PKL)
print(f"  Model loaded successfully from {MODEL_PKL}\n")

# Build one representative sample per fault class from the test set
label_order = sorted(set(y))
print(f"  {'Sample fault_id':<16}  {'Predicted':<16}  Match?")
print(f"  {'-'*16}  {'-'*16}  ------")
all_correct = True
for label in label_order:
    # Pick the first test sample belonging to this class (if any), else train
    indices = np.where(y_test == label)[0]
    if len(indices) == 0:
        indices = np.where(y_train == label)[0]
        src = X_train
    else:
        src = X_test
    sample = src[indices[0]].reshape(1, -1)
    prediction = loaded_model.predict(sample)[0]
    match = "YES" if prediction == label else "NO "
    if prediction != label:
        all_correct = False
    print(f"  {label:<16}  {prediction:<16}  {match}")

if all_correct:
    print("\n  All smoke-test samples predicted correctly.")
else:
    print("\n  WARNING: some smoke-test samples were mis-predicted. Review training data.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
sep("Summary")
print(f"  Model saved     : {MODEL_PKL.name}")
print(f"  Model type      : {best_name}")
print(f"  Test accuracy   : {best_acc*100:.1f}%")
print(f"  Accuracy target : {'MET' if best_acc >= MIN_ACCURACY else 'NOT MET'}")
print(f"  Smoke test      : {'PASSED' if all_correct else 'FAILED'}")
sep()
