"""
engine/diagnoser.py
-------------------
Loads the trained Decision Tree classifier and runs fault prediction.

Given a binary feature vector (produced by normaliser.py) the diagnoser
returns:
  - fault_id        : the predicted fault category label (e.g. "F003")
  - confidence      : the classifier's probability for the top prediction
  - confidence_label: human-readable band ("High" / "Medium" / "Low")
  - all_probabilities: dict mapping each fault_id to its probability score
  - decision_path   : the full decision tree text (export_text) showing the
                      path the model followed — used for UI explainability

The model and metadata are loaded once at module import time so the
Streamlit app does not reload them on every user interaction.

Public API
----------
predict(feature_vector: np.ndarray) -> dict
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import export_text

# ---------------------------------------------------------------------------
# Load model and metadata once at import
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _ROOT / "data" / "model.pkl"
_META_PATH = _ROOT / "data" / "model_meta.json"

with open(_META_PATH, encoding="utf-8") as _f:
    _meta = json.load(_f)

_model = joblib.load(_MODEL_PATH)
_feature_columns: list = _meta["feature_columns"]
_labels: list = _meta["labels"]         # sorted fault ID list, e.g. ["F001",...,"F007"]

# Pre-compute the full decision tree text once (it never changes at runtime)
_FULL_TREE_TEXT: str = export_text(_model, feature_names=_feature_columns)

# ---------------------------------------------------------------------------
# Confidence band thresholds
# ---------------------------------------------------------------------------
_HIGH_THRESHOLD: float = 0.80
_MEDIUM_THRESHOLD: float = 0.50


def _confidence_label(probability: float) -> str:
    if probability >= _HIGH_THRESHOLD:
        return "High"
    elif probability >= _MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def predict(feature_vector: np.ndarray) -> dict:
    """
    Run fault prediction on a binary feature vector.

    Parameters
    ----------
    feature_vector : np.ndarray
        1-D float32 array of shape (n_features,) produced by
        engine.normaliser.build_feature_vector().

    Returns
    -------
    dict with keys:
      "fault_id"          : str   — predicted fault ID, e.g. "F003"
      "confidence"        : float — probability of the top prediction (0.0–1.0)
      "confidence_label"  : str   — "High" / "Medium" / "Low"
      "all_probabilities" : dict  — {fault_id: probability} for all classes
      "decision_path"     : str   — full decision tree text for UI display

    Raises
    ------
    ValueError
        If feature_vector has the wrong shape.
    """
    expected_n = len(_feature_columns)
    if feature_vector.ndim != 1 or feature_vector.shape[0] != expected_n:
        raise ValueError(
            f"feature_vector must be 1-D with {expected_n} elements, "
            f"got shape {feature_vector.shape}"
        )

    sample = feature_vector.reshape(1, -1)

    # Predicted class
    fault_id: str = _model.predict(sample)[0]

    # Class probabilities
    proba = _model.predict_proba(sample)[0]          # shape: (n_classes,)
    classes = list(_model.classes_)                  # class labels in model order

    all_probs: dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    top_confidence: float = float(all_probs[fault_id])

    return {
        "fault_id": fault_id,
        "confidence": top_confidence,
        "confidence_label": _confidence_label(top_confidence),
        "all_probabilities": all_probs,
        "decision_path": _FULL_TREE_TEXT,
    }
