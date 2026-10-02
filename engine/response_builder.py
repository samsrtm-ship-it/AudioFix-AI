"""
engine/response_builder.py
--------------------------
Assembles the final structured result dictionary that the Streamlit UI
consumes and displays.

It merges the output of diagnoser.predict() with the knowledge base record
returned by retriever.get_fault() into a single self-contained result dict.

Public API
----------
build_response(user_input: dict) -> dict
    Full pipeline: normalise → predict → retrieve → assemble.
    This is the single entry point the Streamlit app calls.

build_response_from_vector(feature_vector, fault_record, diagnosis) -> dict
    Lower-level assembler used by tests that supply components directly.
"""

import numpy as np

from engine import diagnoser, normaliser, retriever

# ---------------------------------------------------------------------------
# Severity badge colours (for the Streamlit UI)
# ---------------------------------------------------------------------------
_SEVERITY_COLOUR: dict = {
    "Critical": "#d62728",   # red
    "High":     "#ff7f0e",   # orange
    "Medium":   "#f0c000",   # amber
    "Low":      "#2ca02c",   # green
}

_CONFIDENCE_COLOUR: dict = {
    "High":   "#2ca02c",   # green
    "Medium": "#f0c000",   # amber
    "Low":    "#d62728",   # red
}


# ---------------------------------------------------------------------------
# Public: single entry point
# ---------------------------------------------------------------------------

def build_response(user_input: dict) -> dict:
    """
    Run the complete diagnosis pipeline for a given user input.

    Parameters
    ----------
    user_input : dict
        Raw input from the Streamlit form.  See normaliser.build_feature_vector
        for the expected structure.

    Returns
    -------
    dict — the full structured result (see _assemble for field list).
    """
    # Step 1: normalise
    feature_vector = normaliser.build_feature_vector(user_input)

    # Step 2: predict
    diagnosis = diagnoser.predict(feature_vector)

    # Step 3: retrieve knowledge
    fault_record = retriever.get_fault(diagnosis["fault_id"])

    # Step 4: assemble
    return _assemble(user_input, feature_vector, diagnosis, fault_record)


# ---------------------------------------------------------------------------
# Public: lower-level assembler (useful for unit tests)
# ---------------------------------------------------------------------------

def build_response_from_parts(
    user_input: dict,
    feature_vector: np.ndarray,
    diagnosis: dict,
    fault_record: dict,
) -> dict:
    """
    Assemble a result dict from pre-computed components.
    Used by tests that want to inject specific diagnosis or fault records.
    """
    return _assemble(user_input, feature_vector, diagnosis, fault_record)


# ---------------------------------------------------------------------------
# Internal assembler
# ---------------------------------------------------------------------------

def _assemble(
    user_input: dict,
    feature_vector: np.ndarray,
    diagnosis: dict,
    fault_record: dict,
) -> dict:
    """
    Merge all components into the final result dictionary.

    Result fields
    -------------
    fault_id              str    — e.g. "F003"
    fault_name            str    — e.g. "Distorted Sound"
    severity              str    — "Critical" / "High" / "Medium" / "Low"
    severity_colour       str    — hex colour for UI badge
    escalate              bool   — whether professional service is recommended
    confidence            float  — 0.0 – 1.0
    confidence_label      str    — "High" / "Medium" / "Low"
    confidence_colour     str    — hex colour for UI badge
    all_probabilities     dict   — {fault_id: probability}
    possible_causes       list   — ordered list of cause strings
    troubleshooting_steps list   — ordered list of step strings
    reasoning             str    — plain-language explanation
    decision_path         str    — full decision tree text for explainability
    active_features       list   — feature IDs that were set to 1 (for UI)
    user_input            dict   — echo of the original input
    """
    severity = fault_record.get("severity", "Medium")
    confidence_label = diagnosis["confidence_label"]

    # Which features did the user trigger?
    active_features = [
        col
        for col, val in zip(normaliser.FEATURE_COLUMNS, feature_vector)
        if val == 1.0
    ]

    return {
        # Core identification
        "fault_id":              diagnosis["fault_id"],
        "fault_name":            fault_record["fault_name"],

        # Severity
        "severity":              severity,
        "severity_colour":       _SEVERITY_COLOUR.get(severity, "#888888"),

        # Escalation
        "escalate":              fault_record.get("escalate", False),

        # Confidence
        "confidence":            diagnosis["confidence"],
        "confidence_label":      confidence_label,
        "confidence_colour":     _CONFIDENCE_COLOUR.get(confidence_label, "#888888"),
        "all_probabilities":     diagnosis["all_probabilities"],

        # Knowledge base content
        "possible_causes":       fault_record.get("possible_causes", []),
        "troubleshooting_steps": fault_record.get("troubleshooting_steps", []),
        "reasoning":             fault_record.get("reasoning", ""),

        # Explainability
        "decision_path":         diagnosis["decision_path"],

        # Debug / UI context
        "active_features":       active_features,
        "user_input":            user_input,
    }
