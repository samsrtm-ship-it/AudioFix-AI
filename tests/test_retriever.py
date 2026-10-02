"""
tests/test_retriever.py
-----------------------
Unit tests for engine.retriever.get_fault() and get_all_fault_ids().

Checks:
- get_all_fault_ids() returns all 7 expected fault IDs
- get_fault() returns a dict with all required fields for each fault ID
- Required list fields are non-empty
- Required string fields are non-empty
- escalate is a boolean
- severity is one of the valid values
- Unknown fault_id raises KeyError
"""

import pytest

from engine import retriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_FAULT_IDS = {"F001", "F002", "F003", "F004", "F005", "F006", "F007"}

REQUIRED_FIELDS = {
    "fault_id":              str,
    "fault_name":            str,
    "equipment":             list,
    "primary_symptoms":      list,
    "secondary_symptoms":    list,
    "severity":              str,
    "possible_causes":       list,
    "troubleshooting_steps": list,
    "escalate":              bool,
    "reasoning":             str,
}

VALID_SEVERITY = {"Critical", "High", "Medium", "Low"}


# ---------------------------------------------------------------------------
# get_all_fault_ids
# ---------------------------------------------------------------------------

class TestGetAllFaultIds:

    def test_returns_list(self):
        ids = retriever.get_all_fault_ids()
        assert isinstance(ids, list)

    def test_returns_seven_ids(self):
        ids = retriever.get_all_fault_ids()
        assert len(ids) == 7

    def test_contains_all_expected_ids(self):
        ids = set(retriever.get_all_fault_ids())
        assert ids == EXPECTED_FAULT_IDS

    def test_is_sorted(self):
        ids = retriever.get_all_fault_ids()
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# get_fault — structure checks (parametrised over all 7 fault IDs)
# ---------------------------------------------------------------------------

class TestGetFaultStructure:

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_returns_dict(self, fault_id):
        record = retriever.get_fault(fault_id)
        assert isinstance(record, dict)

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_all_required_fields_present(self, fault_id):
        record = retriever.get_fault(fault_id)
        for field in REQUIRED_FIELDS:
            assert field in record, f"Fault {fault_id}: missing field '{field}'"

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_field_types(self, fault_id):
        record = retriever.get_fault(fault_id)
        for field, expected_type in REQUIRED_FIELDS.items():
            assert isinstance(record[field], expected_type), (
                f"Fault {fault_id}: field '{field}' should be "
                f"{expected_type.__name__}, got {type(record[field]).__name__}"
            )

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_fault_id_matches(self, fault_id):
        record = retriever.get_fault(fault_id)
        assert record["fault_id"] == fault_id

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_fault_name_non_empty(self, fault_id):
        record = retriever.get_fault(fault_id)
        assert len(record["fault_name"].strip()) > 0

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_severity_is_valid(self, fault_id):
        record = retriever.get_fault(fault_id)
        assert record["severity"] in VALID_SEVERITY, (
            f"Fault {fault_id}: invalid severity '{record['severity']}'"
        )

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_possible_causes_non_empty(self, fault_id):
        record = retriever.get_fault(fault_id)
        assert len(record["possible_causes"]) > 0

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_troubleshooting_steps_non_empty(self, fault_id):
        record = retriever.get_fault(fault_id)
        assert len(record["troubleshooting_steps"]) > 0

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_reasoning_non_empty(self, fault_id):
        record = retriever.get_fault(fault_id)
        assert len(record["reasoning"].strip()) > 0

    @pytest.mark.parametrize("fault_id", sorted(EXPECTED_FAULT_IDS))
    def test_equipment_non_empty(self, fault_id):
        record = retriever.get_fault(fault_id)
        assert len(record["equipment"]) > 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestGetFaultErrors:

    def test_unknown_fault_id_raises_key_error(self):
        with pytest.raises(KeyError):
            retriever.get_fault("F999")

    def test_empty_string_raises_key_error(self):
        with pytest.raises(KeyError):
            retriever.get_fault("")
