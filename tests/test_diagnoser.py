"""
tests/test_diagnoser.py
-----------------------
Unit tests for engine.diagnoser.predict().

Checks:
- Returns a dict with all required keys
- Predicted fault_id is one of the known labels
- Confidence is a float in [0, 1]
- Confidence label is one of High / Medium / Low
- all_probabilities contains all 7 fault IDs and sums to 1
- decision_path is a non-empty string
- Each of the 7 canonical symptom vectors predicts its expected fault
- Wrong-shape input raises ValueError
"""

import numpy as np
import pytest

from engine import diagnoser
from engine.normaliser import FEATURE_COLUMNS, build_feature_vector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KNOWN_LABELS = {"F001", "F002", "F003", "F004", "F005", "F006", "F007", "F008"}


def _predict_for(symptoms=None, equipment=None) -> dict:
    """Run predict() for a given symptom/equipment combination."""
    user_input = {
        "symptoms":  symptoms or [],
        "equipment": equipment or [],
    }
    fv = build_feature_vector(user_input)
    return diagnoser.predict(fv)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:

    def test_returns_dict(self):
        result = _predict_for(symptoms=["No sound"], equipment=["Microphone"])
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = _predict_for(symptoms=["No sound"], equipment=["Microphone"])
        for key in ("fault_id", "confidence", "confidence_label",
                    "all_probabilities", "decision_path"):
            assert key in result, f"Missing key: {key}"

    def test_fault_id_is_known_label(self):
        result = _predict_for(symptoms=["No sound"], equipment=["Microphone"])
        assert result["fault_id"] in _KNOWN_LABELS

    def test_confidence_is_float_in_range(self):
        result = _predict_for(symptoms=["Hum or buzz"])
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_label_valid(self):
        result = _predict_for(symptoms=["Distorted sound"])
        assert result["confidence_label"] in ("High", "Medium", "Low")

    def test_all_probabilities_has_all_labels(self):
        result = _predict_for(symptoms=["Acoustic feedback"])
        assert set(result["all_probabilities"].keys()) == _KNOWN_LABELS

    def test_all_probabilities_sum_to_one(self):
        result = _predict_for(symptoms=["Crackling or intermittent sound"])
        total = sum(result["all_probabilities"].values())
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected ~1.0"

    def test_decision_path_is_non_empty_string(self):
        result = _predict_for(symptoms=["No sound"])
        assert isinstance(result["decision_path"], str)
        assert len(result["decision_path"]) > 0

    def test_decision_path_contains_class_keyword(self):
        result = _predict_for(symptoms=["No sound"])
        assert "class:" in result["decision_path"]


# ---------------------------------------------------------------------------
# Canonical fault predictions — one per fault class
# ---------------------------------------------------------------------------

class TestCanonicalPredictions:
    """
    Each test provides the primary symptom for its fault and expects the
    model to return the correct fault_id.  These are the clean, unambiguous
    cases that the model must get right.
    """

    def test_F001_microphone_not_working(self):
        result = _predict_for(
            symptoms=["No sound"],
            equipment=["Microphone"],
        )
        assert result["fault_id"] == "F001"

    def test_F002_low_volume(self):
        result = _predict_for(
            symptoms=["Low volume"],
            equipment=["Mixer", "Speaker"],
        )
        assert result["fault_id"] == "F002"

    def test_F003_distorted_sound(self):
        result = _predict_for(
            symptoms=["Distorted sound"],
            equipment=["Mixer", "Speaker"],
        )
        assert result["fault_id"] == "F003"

    def test_F004_hum_or_buzz(self):
        # F004 is the residual class — no other primary symptom present
        result = _predict_for(
            symptoms=["Hum or buzz"],
            equipment=["Cable", "Mixer"],
        )
        assert result["fault_id"] == "F004"

    def test_F005_acoustic_feedback(self):
        result = _predict_for(
            symptoms=["Acoustic feedback"],
            equipment=["Microphone", "Mixer", "Speaker"],
        )
        assert result["fault_id"] == "F005"

    def test_F006_one_channel_not_working(self):
        result = _predict_for(
            symptoms=["One channel not working"],
            equipment=["Mixer", "Speaker"],
        )
        assert result["fault_id"] == "F006"

    def test_F007_crackling_or_intermittent(self):
        result = _predict_for(
            symptoms=["Crackling or intermittent sound"],
            equipment=["Cable"],
        )
        assert result["fault_id"] == "F007"


# ---------------------------------------------------------------------------
# Confidence — Decision Tree returns probability 1.0 for unambiguous inputs
# ---------------------------------------------------------------------------

class TestConfidence:

    def test_unambiguous_input_is_high_confidence(self):
        result = _predict_for(symptoms=["Acoustic feedback"])
        # Decision Tree should return p=1.0 for a clean single-symptom input
        assert result["confidence_label"] == "High"
        assert result["confidence"] >= 0.80


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_wrong_length_raises_value_error(self):
        bad_vector = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError):
            diagnoser.predict(bad_vector)

    def test_2d_array_raises_value_error(self):
        bad_vector = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32)
        with pytest.raises(ValueError):
            diagnoser.predict(bad_vector)
