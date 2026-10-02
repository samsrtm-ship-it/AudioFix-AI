"""
tests/test_ui_smoke.py
----------------------
Smoke tests for the Streamlit application layer.

These tests do NOT launch a browser or a Streamlit server.
Instead they verify:

1. app.py can be imported as a module without error
   (catches syntax errors and bad imports at load time).

2. The engine integration used by the UI works correctly for every
   possible combination of equipment × symptom that the UI form can
   produce — covering the full input space the Streamlit widgets expose.

3. The helper functions the UI imports from engine.normaliser return
   the expected labels and counts.

4. Edge-case inputs that the UI can legally send to build_response
   do not raise exceptions.
"""

import importlib
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 0. Ensure project root is on sys.path so app.py is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# 1. app.py import smoke test
# ---------------------------------------------------------------------------

class TestAppImport:

    def test_app_module_imports_without_error(self):
        """
        Import app.py as a module.  Streamlit's set_page_config and widget
        calls run at module level but are safe to import in test mode because
        Streamlit stubs out calls when no server is running.
        """
        # Remove cached module if present so we get a fresh import
        sys.modules.pop("app", None)

        # Streamlit raises ScriptRunContext warnings in test mode — suppress them
        import logging
        logging.getLogger("streamlit").setLevel(logging.CRITICAL)

        try:
            import app  # noqa: F401 — we only care that it doesn't raise
        except SystemExit:
            # Streamlit may call sys.exit() during import in some versions
            pass
        except Exception as exc:
            pytest.fail(f"app.py raised an unexpected exception on import: {exc}")


# ---------------------------------------------------------------------------
# 2. Engine integration — full input space the UI can produce
# ---------------------------------------------------------------------------

from engine.normaliser import get_equipment_labels, get_symptom_labels
from engine.response_builder import build_response

EQUIPMENT_LABELS = get_equipment_labels()
SYMPTOM_LABELS   = get_symptom_labels()

# Every single-symptom + equipment combination (35 combinations)
SINGLE_SYMPTOM_CASES = [
    (symptom, equipment)
    for symptom   in SYMPTOM_LABELS
    for equipment in EQUIPMENT_LABELS
]


@pytest.mark.parametrize("symptom,equipment", SINGLE_SYMPTOM_CASES,
                          ids=[f"{s[:12]}/{e}" for s, e in SINGLE_SYMPTOM_CASES])
def test_single_symptom_per_equipment(symptom, equipment):
    """Each symptom × equipment combination returns a valid result."""
    result = build_response({"symptoms": [symptom], "equipment": [equipment]})

    assert "fault_id" in result
    assert result["fault_id"] in {f"F{i:03d}" for i in range(1, 8)}
    assert len(result["fault_name"]) > 0
    assert len(result["possible_causes"]) > 0
    assert len(result["troubleshooting_steps"]) > 0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence_label"] in ("High", "Medium", "Low")
    assert result["severity"] in ("Critical", "High", "Medium", "Low")
    assert isinstance(result["escalate"], bool)
    assert len(result["decision_path"]) > 0


class TestUIEdgeCases:

    def test_no_symptoms_no_equipment(self):
        """Empty form submission must not raise."""
        result = build_response({"symptoms": [], "equipment": []})
        assert "fault_id" in result

    def test_no_symptoms_with_equipment(self):
        """Equipment selected but no symptom — should not raise."""
        result = build_response({"symptoms": [], "equipment": ["Microphone"]})
        assert "fault_id" in result

    def test_all_symptoms_all_equipment(self):
        """Selecting everything at once must not raise."""
        result = build_response({
            "symptoms":  SYMPTOM_LABELS,
            "equipment": EQUIPMENT_LABELS,
        })
        assert "fault_id" in result

    def test_multiple_symptoms_selected(self):
        """Two symptoms selected simultaneously."""
        result = build_response({
            "symptoms":  ["No sound", "Hum or buzz"],
            "equipment": ["Microphone", "Cable"],
        })
        assert "fault_id" in result

    def test_extra_context_fields_do_not_raise(self):
        """
        The UI sends extra context fields (when_started, connection_type, etc.)
        that the engine does not use — they must be silently ignored.
        """
        result = build_response({
            "symptoms":        ["Distorted sound"],
            "equipment":       ["Mixer"],
            "when_started":    "Suddenly",
            "connection_type": "XLR",
            "power_status":    "Powered on (normal)",
            "environment":     "Indoors – classroom / studio",
            "notes":           "Already replaced the cable.",
        })
        assert result["fault_id"] == "F003"

    def test_result_has_all_ui_fields(self):
        """Every field the UI reads must be present in the result dict."""
        result = build_response({"symptoms": ["Acoustic feedback"], "equipment": ["Speaker"]})
        ui_fields = [
            "fault_id", "fault_name", "severity", "severity_colour",
            "escalate", "confidence", "confidence_label", "confidence_colour",
            "all_probabilities", "possible_causes", "troubleshooting_steps",
            "reasoning", "decision_path", "active_features", "user_input",
        ]
        for field in ui_fields:
            assert field in result, f"Field '{field}' missing from result"

    def test_all_probabilities_has_seven_entries(self):
        """Probability chart needs exactly 7 entries."""
        result = build_response({"symptoms": ["Low volume"], "equipment": ["Speaker"]})
        assert len(result["all_probabilities"]) == 7

    def test_active_features_is_list(self):
        result = build_response({"symptoms": ["Crackling or intermittent sound"], "equipment": ["Cable"]})
        assert isinstance(result["active_features"], list)


# ---------------------------------------------------------------------------
# 3. Normaliser label helpers used by the UI
# ---------------------------------------------------------------------------

class TestNormaliserHelpers:

    def test_get_symptom_labels_count(self):
        assert len(SYMPTOM_LABELS) == 7

    def test_get_equipment_labels_count(self):
        assert len(EQUIPMENT_LABELS) == 5

    def test_symptom_labels_are_strings(self):
        assert all(isinstance(s, str) for s in SYMPTOM_LABELS)

    def test_equipment_labels_are_strings(self):
        assert all(isinstance(e, str) for e in EQUIPMENT_LABELS)

    def test_no_sound_in_symptom_labels(self):
        assert "No sound" in SYMPTOM_LABELS

    def test_microphone_in_equipment_labels(self):
        assert "Microphone" in EQUIPMENT_LABELS
