"""
tests/test_normaliser.py
------------------------
Unit tests for engine.normaliser.build_feature_vector().

Checks:
- Correct vector length and dtype
- Correct feature positions for all 7 symptom labels
- Correct feature positions for all 5 equipment labels
- Duplicate entries do not cause values > 1
- Unknown inputs are silently ignored
- Internal feature IDs are accepted as well as display labels
- Empty input produces all-zero vector
- Non-list equipment string is handled
- ValueError is raised for non-dict input
"""

import numpy as np
import pytest

from engine.normaliser import (
    FEATURE_COLUMNS,
    VALID_EQUIPMENT,
    VALID_SYMPTOMS,
    build_feature_vector,
    get_equipment_labels,
    get_symptom_labels,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _idx(feature_id: str) -> int:
    """Return the index of a feature column by its ID."""
    return FEATURE_COLUMNS.index(feature_id)


def _vec(**kwargs) -> np.ndarray:
    """Shorthand: build_feature_vector({"symptoms": [...], "equipment": [...]})."""
    return build_feature_vector(kwargs)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestVectorStructure:

    def test_length_matches_feature_columns(self):
        v = build_feature_vector({"symptoms": [], "equipment": []})
        assert len(v) == len(FEATURE_COLUMNS)

    def test_dtype_is_float32(self):
        v = build_feature_vector({"symptoms": [], "equipment": []})
        assert v.dtype == np.float32

    def test_empty_input_all_zeros(self):
        v = build_feature_vector({"symptoms": [], "equipment": []})
        assert np.all(v == 0.0)

    def test_missing_symptoms_key_defaults_to_zeros(self):
        v = build_feature_vector({"equipment": ["Microphone"]})
        for sym_id in VALID_SYMPTOMS:
            assert v[_idx(sym_id)] == 0.0

    def test_missing_equipment_key_defaults_to_zeros(self):
        v = build_feature_vector({"symptoms": ["No sound"]})
        for eq_id in VALID_EQUIPMENT:
            assert v[_idx(eq_id)] == 0.0

    def test_non_dict_raises_value_error(self):
        with pytest.raises(ValueError):
            build_feature_vector(["No sound"])


# ---------------------------------------------------------------------------
# Symptom label → feature ID mapping
# ---------------------------------------------------------------------------

class TestSymptomMapping:

    @pytest.mark.parametrize("label,feature_id", [
        ("No sound",                        "symptom_no_sound"),
        ("Low volume",                      "symptom_low_volume"),
        ("Distorted sound",                 "symptom_distortion"),
        ("Hum or buzz",                     "symptom_hum_buzz"),
        ("Acoustic feedback",               "symptom_feedback"),
        ("One channel not working",         "symptom_one_channel"),
        ("Crackling or intermittent sound", "symptom_crackling"),
    ])
    def test_display_label_sets_correct_bit(self, label, feature_id):
        v = build_feature_vector({"symptoms": [label], "equipment": []})
        assert v[_idx(feature_id)] == 1.0

    @pytest.mark.parametrize("label,feature_id", [
        ("No sound",                        "symptom_no_sound"),
        ("Low volume",                      "symptom_low_volume"),
        ("Distorted sound",                 "symptom_distortion"),
        ("Hum or buzz",                     "symptom_hum_buzz"),
        ("Acoustic feedback",               "symptom_feedback"),
        ("One channel not working",         "symptom_one_channel"),
        ("Crackling or intermittent sound", "symptom_crackling"),
    ])
    def test_only_selected_symptom_bit_is_set(self, label, feature_id):
        v = build_feature_vector({"symptoms": [label], "equipment": []})
        for col in VALID_SYMPTOMS:
            expected = 1.0 if col == feature_id else 0.0
            assert v[_idx(col)] == expected, f"Expected {col}={expected}, got {v[_idx(col)]}"

    def test_internal_feature_id_accepted(self):
        v = build_feature_vector({"symptoms": ["symptom_no_sound"], "equipment": []})
        assert v[_idx("symptom_no_sound")] == 1.0

    def test_multiple_symptoms_all_set(self):
        v = build_feature_vector({
            "symptoms": ["No sound", "Hum or buzz"],
            "equipment": [],
        })
        assert v[_idx("symptom_no_sound")] == 1.0
        assert v[_idx("symptom_hum_buzz")] == 1.0

    def test_unknown_symptom_ignored(self):
        v = build_feature_vector({"symptoms": ["UNKNOWN_SYMPTOM"], "equipment": []})
        assert np.all(v == 0.0)

    def test_duplicate_symptom_does_not_exceed_one(self):
        v = build_feature_vector({
            "symptoms": ["No sound", "No sound"],
            "equipment": [],
        })
        assert v[_idx("symptom_no_sound")] == 1.0


# ---------------------------------------------------------------------------
# Equipment label → feature ID mapping
# ---------------------------------------------------------------------------

class TestEquipmentMapping:

    @pytest.mark.parametrize("label,feature_id", [
        ("Microphone", "equip_microphone"),
        ("Mixer",      "equip_mixer"),
        ("Speaker",    "equip_speaker"),
        ("Cable",      "equip_cable"),
        ("Amplifier",  "equip_amplifier"),
    ])
    def test_display_label_sets_correct_bit(self, label, feature_id):
        v = build_feature_vector({"symptoms": [], "equipment": [label]})
        assert v[_idx(feature_id)] == 1.0

    def test_string_equipment_handled(self):
        """Single string (not a list) should be handled gracefully."""
        v = build_feature_vector({"symptoms": [], "equipment": "Microphone"})
        assert v[_idx("equip_microphone")] == 1.0

    def test_internal_equipment_id_accepted(self):
        v = build_feature_vector({"symptoms": [], "equipment": ["equip_cable"]})
        assert v[_idx("equip_cable")] == 1.0

    def test_unknown_equipment_ignored(self):
        v = build_feature_vector({"symptoms": [], "equipment": ["TURNTABLE"]})
        for eq_id in VALID_EQUIPMENT:
            assert v[_idx(eq_id)] == 0.0


# ---------------------------------------------------------------------------
# Combined inputs
# ---------------------------------------------------------------------------

class TestCombinedInput:

    def test_symptom_and_equipment_both_set(self):
        v = build_feature_vector({
            "symptoms":  ["No sound"],
            "equipment": ["Microphone"],
        })
        assert v[_idx("symptom_no_sound")] == 1.0
        assert v[_idx("equip_microphone")] == 1.0

    def test_all_symptoms_all_equipment(self):
        v = build_feature_vector({
            "symptoms":  get_symptom_labels(),
            "equipment": get_equipment_labels(),
        })
        for col in FEATURE_COLUMNS:
            assert v[_idx(col)] == 1.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_get_symptom_labels_returns_seven(self):
        assert len(get_symptom_labels()) == 7

    def test_get_equipment_labels_returns_five(self):
        assert len(get_equipment_labels()) == 5
