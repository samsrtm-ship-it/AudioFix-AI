"""
tests/test_response_builder.py
-------------------------------
Unit tests for engine.response_builder.build_response().

Checks:
- Returns a dict with all required output fields
- Each field has the correct type
- Severity and confidence colours are non-empty hex strings
- active_features only contains recognised feature column names
- escalate is a boolean
- All 7 canonical fault scenarios produce the correct fault_id
- Edge case: no symptoms selected returns a valid (low-confidence) result
"""

import pytest

from engine import response_builder
from engine.normaliser import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "fault_id":              str,
    "fault_name":            str,
    "severity":              str,
    "severity_colour":       str,
    "escalate":              bool,
    "confidence":            float,
    "confidence_label":      str,
    "confidence_colour":     str,
    "all_probabilities":     dict,
    "possible_causes":       list,
    "troubleshooting_steps": list,
    "reasoning":             str,
    "decision_path":         str,
    "active_features":       list,
    "user_input":            dict,
}


def _diagnose(symptoms=None, equipment=None) -> dict:
    return response_builder.build_response({
        "symptoms":  symptoms or [],
        "equipment": equipment or [],
    })


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:

    def test_returns_dict(self):
        result = _diagnose(symptoms=["No sound"], equipment=["Microphone"])
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        result = _diagnose(symptoms=["No sound"], equipment=["Microphone"])
        for key in REQUIRED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    def test_field_types(self):
        result = _diagnose(symptoms=["Distorted sound"], equipment=["Mixer"])
        for key, expected_type in REQUIRED_KEYS.items():
            assert isinstance(result[key], expected_type), (
                f"Field '{key}' should be {expected_type.__name__}, "
                f"got {type(result[key]).__name__}"
            )

    def test_severity_colour_is_hex(self):
        result = _diagnose(symptoms=["Acoustic feedback"])
        colour = result["severity_colour"]
        assert colour.startswith("#"), f"severity_colour '{colour}' is not a hex colour"
        assert len(colour) == 7

    def test_confidence_colour_is_hex(self):
        result = _diagnose(symptoms=["Hum or buzz"])
        colour = result["confidence_colour"]
        assert colour.startswith("#")
        assert len(colour) == 7

    def test_active_features_are_valid_columns(self):
        result = _diagnose(
            symptoms=["No sound", "Hum or buzz"],
            equipment=["Microphone", "Cable"],
        )
        for feat in result["active_features"]:
            assert feat in FEATURE_COLUMNS, f"Unknown feature in active_features: '{feat}'"

    def test_user_input_echoed(self):
        inp = {"symptoms": ["Low volume"], "equipment": ["Speaker"]}
        result = response_builder.build_response(inp)
        assert result["user_input"] == inp

    def test_possible_causes_non_empty(self):
        result = _diagnose(symptoms=["Crackling or intermittent sound"])
        assert len(result["possible_causes"]) > 0

    def test_troubleshooting_steps_non_empty(self):
        result = _diagnose(symptoms=["One channel not working"])
        assert len(result["troubleshooting_steps"]) > 0

    def test_reasoning_non_empty(self):
        result = _diagnose(symptoms=["Distorted sound"])
        assert len(result["reasoning"].strip()) > 0

    def test_decision_path_non_empty(self):
        result = _diagnose(symptoms=["No sound"])
        assert len(result["decision_path"]) > 0

    def test_escalate_is_bool(self):
        result = _diagnose(symptoms=["No sound"])
        assert isinstance(result["escalate"], bool)


# ---------------------------------------------------------------------------
# Canonical fault scenarios — all 7 faults
# ---------------------------------------------------------------------------

class TestCanonicalFaults:

    @pytest.mark.parametrize("symptoms,equipment,expected_fault", [
        (["No sound"],                        ["Microphone"],          "F001"),
        (["Low volume"],                      ["Mixer", "Speaker"],    "F002"),
        (["Distorted sound"],                 ["Mixer", "Speaker"],    "F003"),
        (["Hum or buzz"],                     ["Cable", "Mixer"],      "F004"),
        (["Acoustic feedback"],               ["Microphone", "Speaker"], "F005"),
        (["One channel not working"],         ["Mixer", "Speaker"],    "F006"),
        (["Crackling or intermittent sound"], ["Cable"],               "F007"),
    ])
    def test_fault_id_correct(self, symptoms, equipment, expected_fault):
        result = _diagnose(symptoms=symptoms, equipment=equipment)
        assert result["fault_id"] == expected_fault, (
            f"Expected {expected_fault}, got {result['fault_id']} "
            f"for symptoms={symptoms}"
        )

    @pytest.mark.parametrize("symptoms,equipment,expected_fault", [
        (["No sound"],                        ["Microphone"],          "F001"),
        (["Low volume"],                      ["Mixer", "Speaker"],    "F002"),
        (["Distorted sound"],                 ["Mixer", "Speaker"],    "F003"),
        (["Hum or buzz"],                     ["Cable", "Mixer"],      "F004"),
        (["Acoustic feedback"],               ["Microphone", "Speaker"], "F005"),
        (["One channel not working"],         ["Mixer", "Speaker"],    "F006"),
        (["Crackling or intermittent sound"], ["Cable"],               "F007"),
    ])
    def test_fault_name_non_empty(self, symptoms, equipment, expected_fault):
        result = _diagnose(symptoms=symptoms, equipment=equipment)
        assert len(result["fault_name"].strip()) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_no_symptoms_returns_valid_result(self):
        """Empty input should not raise — engine must handle it gracefully."""
        result = _diagnose(symptoms=[], equipment=[])
        assert "fault_id" in result
        assert result["fault_id"] in {f"F{i:03d}" for i in range(1, 8)}

    def test_all_symptoms_returns_valid_result(self):
        from engine.normaliser import get_symptom_labels, get_equipment_labels
        result = _diagnose(
            symptoms=get_symptom_labels(),
            equipment=get_equipment_labels(),
        )
        assert "fault_id" in result
