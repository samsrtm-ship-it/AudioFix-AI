"""
engine/normaliser.py
--------------------
Converts the raw user-input dictionary (from the Streamlit form) into a
binary NumPy feature vector that matches the column order expected by the
trained Decision Tree classifier.

The feature column order is loaded from data/model_meta.json at import time,
so the normaliser is always in sync with the model even if column order
changes across training runs.

Public API
----------
build_feature_vector(user_input: dict) -> np.ndarray
    Returns a 1-D float32 NumPy array of shape (n_features,).

VALID_SYMPTOMS : list[str]
    Ordered list of symptom feature IDs (e.g. "symptom_no_sound").

VALID_EQUIPMENT : list[str]
    Ordered list of equipment feature IDs (e.g. "equip_microphone").

FEATURE_COLUMNS : list[str]
    Full ordered feature column list loaded from model_meta.json.
"""

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load feature metadata from model_meta.json
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_META_PATH = _ROOT / "data" / "model_meta.json"

with open(_META_PATH, encoding="utf-8") as _f:
    _meta = json.load(_f)

FEATURE_COLUMNS: list = _meta["feature_columns"]

# Split feature columns into symptom and equipment buckets for validation
VALID_SYMPTOMS: list = [c for c in FEATURE_COLUMNS if c.startswith("symptom_")]
VALID_EQUIPMENT: list = [c for c in FEATURE_COLUMNS if c.startswith("equip_")]

# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

# Maps UI-visible symptom labels to internal feature IDs
_SYMPTOM_LABEL_TO_FEATURE: dict = {
    "No sound":                       "symptom_no_sound",
    "Low volume":                     "symptom_low_volume",
    "Distorted sound":                "symptom_distortion",
    "Hum or buzz":                    "symptom_hum_buzz",
    "Acoustic feedback":              "symptom_feedback",
    "One channel not working":        "symptom_one_channel",
    "Crackling or intermittent sound": "symptom_crackling",
}

# Maps UI-visible equipment labels to internal feature IDs
_EQUIPMENT_LABEL_TO_FEATURE: dict = {
    "Microphone": "equip_microphone",
    "Mixer":      "equip_mixer",
    "Speaker":    "equip_speaker",
    "Cable":      "equip_cable",
    "Amplifier":  "equip_amplifier",
}


def get_symptom_labels() -> list:
    """Return the ordered list of symptom display labels for the UI."""
    return list(_SYMPTOM_LABEL_TO_FEATURE.keys())


def get_equipment_labels() -> list:
    """Return the ordered list of equipment display labels for the UI."""
    return list(_EQUIPMENT_LABEL_TO_FEATURE.keys())


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def build_feature_vector(user_input: dict) -> np.ndarray:
    """
    Convert the raw user-input dictionary into a binary feature vector.

    Parameters
    ----------
    user_input : dict
        Expected keys:
          "symptoms"  : list[str]  — symptom display labels selected by the user
                        OR symptom feature IDs (both are accepted)
          "equipment" : str | list[str]  — equipment display label(s) selected
                        OR equipment feature IDs (both are accepted)

        Both display labels (e.g. "No sound") and internal feature IDs
        (e.g. "symptom_no_sound") are accepted so the normaliser can be used
        directly from tests without needing the full UI label strings.

    Returns
    -------
    np.ndarray
        1-D float32 array of shape (len(FEATURE_COLUMNS),) with values 0.0
        or 1.0 in the exact column order of FEATURE_COLUMNS.

    Raises
    ------
    ValueError
        If user_input is missing required keys or is otherwise malformed.
    """
    if not isinstance(user_input, dict):
        raise ValueError("user_input must be a dict")

    # ---- symptoms ----------------------------------------------------------
    raw_symptoms = user_input.get("symptoms", [])
    if isinstance(raw_symptoms, str):
        raw_symptoms = [raw_symptoms]

    active_symptom_features: set = set()
    for sym in raw_symptoms:
        # Accept either display label or feature ID
        if sym in _SYMPTOM_LABEL_TO_FEATURE:
            active_symptom_features.add(_SYMPTOM_LABEL_TO_FEATURE[sym])
        elif sym in VALID_SYMPTOMS:
            active_symptom_features.add(sym)
        # Unknown values are silently ignored (graceful degradation)

    # ---- equipment ---------------------------------------------------------
    raw_equipment = user_input.get("equipment", [])
    if isinstance(raw_equipment, str):
        raw_equipment = [raw_equipment]

    active_equipment_features: set = set()
    for eq in raw_equipment:
        if eq in _EQUIPMENT_LABEL_TO_FEATURE:
            active_equipment_features.add(_EQUIPMENT_LABEL_TO_FEATURE[eq])
        elif eq in VALID_EQUIPMENT:
            active_equipment_features.add(eq)

    # ---- build vector in FEATURE_COLUMNS order -----------------------------
    vector = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLUMNS):
        if col in active_symptom_features or col in active_equipment_features:
            vector[i] = 1.0

    return vector
