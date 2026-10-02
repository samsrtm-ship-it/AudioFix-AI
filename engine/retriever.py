"""
engine/retriever.py
-------------------
Looks up a full fault record from the knowledge base (data/faults.json)
by fault ID and returns it as a dict.

The knowledge base is loaded once at module import time.

Public API
----------
get_fault(fault_id: str) -> dict
    Returns the matching fault record.

get_all_fault_ids() -> list[str]
    Returns a sorted list of all fault IDs in the knowledge base.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Load knowledge base once at import
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_FAULTS_PATH = _ROOT / "data" / "faults.json"

with open(_FAULTS_PATH, encoding="utf-8") as _f:
    _faults_list: list = json.load(_f)

# Index by fault_id for O(1) lookup
_faults_index: dict = {record["fault_id"]: record for record in _faults_list}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_fault(fault_id: str) -> dict:
    """
    Return the full knowledge base record for a given fault ID.

    Parameters
    ----------
    fault_id : str
        The fault ID to look up, e.g. "F003".

    Returns
    -------
    dict
        The matching fault record containing:
          fault_id, fault_name, equipment, primary_symptoms,
          secondary_symptoms, severity, possible_causes,
          troubleshooting_steps, escalate, reasoning

    Raises
    ------
    KeyError
        If fault_id is not found in the knowledge base.
    """
    if fault_id not in _faults_index:
        raise KeyError(
            f"Fault ID '{fault_id}' not found in knowledge base. "
            f"Available IDs: {sorted(_faults_index.keys())}"
        )
    return _faults_index[fault_id]


def get_all_fault_ids() -> list:
    """Return a sorted list of all fault IDs in the knowledge base."""
    return sorted(_faults_index.keys())
