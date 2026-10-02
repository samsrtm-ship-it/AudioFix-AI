"""
tests/test_integration.py
-------------------------
End-to-end integration test for the full engine pipeline:

    raw user_input dict
        → normaliser.build_feature_vector()
        → diagnoser.predict()
        → retriever.get_fault()
        → response_builder.build_response()
        → final result dict

Tests all 7 fault scenarios end-to-end and verifies that the complete
result dict is fully populated with correct types and non-empty content.
"""

import pytest

from engine.response_builder import build_response

# ---------------------------------------------------------------------------
# Full pipeline scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "description":    "F001 — Microphone Not Working",
        "user_input":     {"symptoms": ["No sound"], "equipment": ["Microphone"]},
        "expected_fault": "F001",
    },
    {
        "description":    "F002 — Low Volume",
        "user_input":     {"symptoms": ["Low volume"], "equipment": ["Mixer", "Speaker"]},
        "expected_fault": "F002",
    },
    {
        "description":    "F003 — Distorted Sound",
        "user_input":     {"symptoms": ["Distorted sound"], "equipment": ["Mixer"]},
        "expected_fault": "F003",
    },
    {
        "description":    "F004 — Hum or Buzz",
        "user_input":     {"symptoms": ["Hum or buzz"], "equipment": ["Cable", "Mixer"]},
        "expected_fault": "F004",
    },
    {
        "description":    "F005 — Acoustic Feedback",
        "user_input":     {"symptoms": ["Acoustic feedback"],
                           "equipment": ["Microphone", "Mixer", "Speaker"]},
        "expected_fault": "F005",
    },
    {
        "description":    "F006 — One Channel Not Working",
        "user_input":     {"symptoms": ["One channel not working"],
                           "equipment": ["Mixer", "Speaker"]},
        "expected_fault": "F006",
    },
    {
        "description":    "F007 — Crackling or Intermittent Sound",
        "user_input":     {"symptoms": ["Crackling or intermittent sound"],
                           "equipment": ["Cable"]},
        "expected_fault": "F007",
    },
]

# All required top-level keys in the final result
REQUIRED_KEYS = [
    "fault_id", "fault_name", "severity", "severity_colour",
    "escalate", "confidence", "confidence_label", "confidence_colour",
    "all_probabilities", "possible_causes", "troubleshooting_steps",
    "reasoning", "decision_path", "active_features", "user_input",
]


# ---------------------------------------------------------------------------
# Parametrised end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["description"] for s in SCENARIOS])
def test_pipeline_correct_fault(scenario):
    """Full pipeline returns the correct fault_id for each canonical scenario."""
    result = build_response(scenario["user_input"])
    assert result["fault_id"] == scenario["expected_fault"], (
        f"{scenario['description']}: expected {scenario['expected_fault']}, "
        f"got {result['fault_id']}"
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["description"] for s in SCENARIOS])
def test_pipeline_all_keys_present(scenario):
    """Full pipeline result contains all required keys."""
    result = build_response(scenario["user_input"])
    for key in REQUIRED_KEYS:
        assert key in result, f"{scenario['description']}: missing key '{key}'"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["description"] for s in SCENARIOS])
def test_pipeline_knowledge_base_content_populated(scenario):
    """Knowledge base fields are populated and non-empty."""
    result = build_response(scenario["user_input"])
    assert len(result["fault_name"].strip()) > 0,          "fault_name empty"
    assert len(result["possible_causes"]) > 0,             "possible_causes empty"
    assert len(result["troubleshooting_steps"]) > 0,       "troubleshooting_steps empty"
    assert len(result["reasoning"].strip()) > 0,           "reasoning empty"
    assert len(result["decision_path"].strip()) > 0,       "decision_path empty"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["description"] for s in SCENARIOS])
def test_pipeline_confidence_in_range(scenario):
    """Confidence score is a float in [0, 1]."""
    result = build_response(scenario["user_input"])
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["description"] for s in SCENARIOS])
def test_pipeline_probabilities_sum_to_one(scenario):
    """All class probabilities sum to 1.0."""
    result = build_response(scenario["user_input"])
    total = sum(result["all_probabilities"].values())
    assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["description"] for s in SCENARIOS])
def test_pipeline_user_input_echoed(scenario):
    """user_input field in result matches the original input dict."""
    result = build_response(scenario["user_input"])
    assert result["user_input"] == scenario["user_input"]


# ---------------------------------------------------------------------------
# Edge case: empty input
# ---------------------------------------------------------------------------

def test_pipeline_empty_input_does_not_raise():
    """Empty symptoms and equipment must not raise an exception."""
    result = build_response({"symptoms": [], "equipment": []})
    assert "fault_id" in result
    assert result["fault_id"] in {f"F{i:03d}" for i in range(1, 8)}


def test_pipeline_extra_context_fields_ignored():
    """
    The engine ignores unknown extra keys in user_input
    (e.g. when_started, connection_type, environment).
    """
    user_input = {
        "symptoms":        ["No sound"],
        "equipment":       ["Microphone"],
        "when_started":    "Suddenly",
        "connection_type": "XLR",
        "environment":     "Studio",
        "notes":           "Tried replacing the cable already",
    }
    result = build_response(user_input)
    assert result["fault_id"] == "F001"
