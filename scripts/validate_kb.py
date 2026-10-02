"""
validate_kb.py
--------------
Validates that data/faults.json and data/symptoms.json are complete,
consistent, and schema-compliant.

Also validates that data/training_data.csv columns match the feature IDs
defined in symptoms.json and that all fault_id values in the CSV exist
in faults.json.

Run from the project root:
    python scripts/validate_kb.py
"""

import json
import csv
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FAULTS_PATH = ROOT / "data" / "faults.json"
SYMPTOMS_PATH = ROOT / "data" / "symptoms.json"
TRAINING_PATH = ROOT / "data" / "training_data.csv"

# ── Required fields ────────────────────────────────────────────────────────────
FAULT_REQUIRED_FIELDS = {
    "fault_id": str,
    "fault_name": str,
    "equipment": list,
    "primary_symptoms": list,
    "secondary_symptoms": list,
    "severity": str,
    "possible_causes": list,
    "troubleshooting_steps": list,
    "escalate": bool,
    "reasoning": str,
}

FAULT_SEVERITY_VALUES = {"Critical", "High", "Medium", "Low"}

SYMPTOM_REQUIRED_FIELDS = {
    "symptom_id": str,
    "label": str,
    "feature_id": str,
    "description": str,
    "common_equipment": list,
}

EXPECTED_FAULT_COUNT = 7
EXPECTED_SYMPTOM_COUNT = 7
MIN_TRAINING_ROWS_PER_FAULT = 15

# ── Helpers ────────────────────────────────────────────────────────────────────
errors = []
warnings = []


def err(msg: str) -> None:
    errors.append(f"  ERROR: {msg}")


def warn(msg: str) -> None:
    warnings.append(f"  WARN : {msg}")


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# ── 1. Validate faults.json ────────────────────────────────────────────────────
section("1. Validating data/faults.json")

if not FAULTS_PATH.exists():
    err(f"File not found: {FAULTS_PATH}")
    print("\n".join(errors))
    sys.exit(1)

with open(FAULTS_PATH, encoding="utf-8") as f:
    faults = json.load(f)

print(f"  Loaded {len(faults)} fault record(s).")

if len(faults) != EXPECTED_FAULT_COUNT:
    err(f"Expected {EXPECTED_FAULT_COUNT} fault records, found {len(faults)}.")

fault_ids_seen = set()

for i, fault in enumerate(faults):
    prefix = f"faults[{i}] ({fault.get('fault_id', '?')})"

    # Required fields
    for field, expected_type in FAULT_REQUIRED_FIELDS.items():
        if field not in fault:
            err(f"{prefix}: missing required field '{field}'")
        elif not isinstance(fault[field], expected_type):
            err(
                f"{prefix}: field '{field}' should be {expected_type.__name__}, "
                f"got {type(fault[field]).__name__}"
            )

    # Duplicate fault_id check
    fid = fault.get("fault_id", "")
    if fid in fault_ids_seen:
        err(f"{prefix}: duplicate fault_id '{fid}'")
    fault_ids_seen.add(fid)

    # Severity value
    severity = fault.get("severity", "")
    if severity not in FAULT_SEVERITY_VALUES:
        err(
            f"{prefix}: invalid severity '{severity}'. "
            f"Must be one of {sorted(FAULT_SEVERITY_VALUES)}"
        )

    # Non-empty lists
    for list_field in ("equipment", "primary_symptoms", "possible_causes", "troubleshooting_steps"):
        if field in fault and isinstance(fault.get(list_field), list):
            if len(fault.get(list_field, [])) == 0:
                err(f"{prefix}: '{list_field}' must not be empty")

    # Reasoning non-empty
    if isinstance(fault.get("reasoning"), str) and len(fault.get("reasoning", "").strip()) == 0:
        err(f"{prefix}: 'reasoning' must not be an empty string")

    # Troubleshooting steps minimum
    steps = fault.get("troubleshooting_steps", [])
    if isinstance(steps, list) and len(steps) < 3:
        warn(f"{prefix}: only {len(steps)} troubleshooting step(s) — recommend at least 3")

print(f"  Fault IDs found: {sorted(fault_ids_seen)}")

# ── 2. Validate symptoms.json ──────────────────────────────────────────────────
section("2. Validating data/symptoms.json")

if not SYMPTOMS_PATH.exists():
    err(f"File not found: {SYMPTOMS_PATH}")
else:
    with open(SYMPTOMS_PATH, encoding="utf-8") as f:
        symptoms = json.load(f)

    print(f"  Loaded {len(symptoms)} symptom record(s).")

    if len(symptoms) != EXPECTED_SYMPTOM_COUNT:
        err(f"Expected {EXPECTED_SYMPTOM_COUNT} symptom records, found {len(symptoms)}.")

    symptom_ids_seen = set()
    feature_ids_seen = set()

    for i, symptom in enumerate(symptoms):
        prefix = f"symptoms[{i}] ({symptom.get('symptom_id', '?')})"

        for field, expected_type in SYMPTOM_REQUIRED_FIELDS.items():
            if field not in symptom:
                err(f"{prefix}: missing required field '{field}'")
            elif not isinstance(symptom[field], expected_type):
                err(
                    f"{prefix}: field '{field}' should be {expected_type.__name__}, "
                    f"got {type(symptom[field]).__name__}"
                )

        sid = symptom.get("symptom_id", "")
        if sid in symptom_ids_seen:
            err(f"{prefix}: duplicate symptom_id '{sid}'")
        symptom_ids_seen.add(sid)

        fid = symptom.get("feature_id", "")
        if fid in feature_ids_seen:
            err(f"{prefix}: duplicate feature_id '{fid}'")
        feature_ids_seen.add(fid)

        if not fid.startswith("symptom_"):
            warn(f"{prefix}: feature_id '{fid}' does not follow 'symptom_*' naming convention")

    print(f"  Symptom feature IDs: {sorted(feature_ids_seen)}")

# ── 3. Validate training_data.csv ──────────────────────────────────────────────
section("3. Validating data/training_data.csv")

if not TRAINING_PATH.exists():
    err(f"File not found: {TRAINING_PATH}")
else:
    with open(TRAINING_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        csv_columns = reader.fieldnames or []

    print(f"  Loaded {len(rows)} training row(s) (excluding header).")

    # Expected feature columns derived from symptoms.json feature_ids
    expected_symptom_cols = sorted(feature_ids_seen) if "feature_ids_seen" in dir() else []
    expected_equip_cols = [
        "equip_amplifier", "equip_cable", "equip_microphone", "equip_mixer", "equip_speaker"
    ]
    expected_cols = sorted(expected_symptom_cols + expected_equip_cols) + ["fault_id"]

    # Check all expected columns present
    for col in expected_cols:
        if col not in csv_columns:
            err(f"training_data.csv: missing expected column '{col}'")

    # Check no unexpected columns
    for col in csv_columns:
        if col not in expected_cols:
            warn(f"training_data.csv: unexpected column '{col}'")

    # Count rows per fault and validate values
    fault_counts: dict = {}
    for row_num, row in enumerate(rows, start=2):  # row 1 = header
        fid = row.get("fault_id", "").strip()

        if fid not in fault_ids_seen:
            err(f"training_data.csv row {row_num}: unknown fault_id '{fid}'")

        fault_counts[fid] = fault_counts.get(fid, 0) + 1

        # All feature columns must be 0 or 1
        for col in csv_columns:
            if col == "fault_id":
                continue
            val = row.get(col, "").strip()
            if val not in ("0", "1"):
                err(
                    f"training_data.csv row {row_num}, col '{col}': "
                    f"expected 0 or 1, got '{val}'"
                )

    print(f"  Rows per fault category:")
    for fid in sorted(fault_counts):
        count = fault_counts[fid]
        status = "OK" if count >= MIN_TRAINING_ROWS_PER_FAULT else "LOW"
        print(f"    {fid}: {count} rows  [{status}]")
        if count < MIN_TRAINING_ROWS_PER_FAULT:
            err(
                f"training_data.csv: fault '{fid}' has only {count} training row(s); "
                f"minimum is {MIN_TRAINING_ROWS_PER_FAULT}"
            )

    # Check all fault IDs from faults.json have training rows
    for fid in fault_ids_seen:
        if fid not in fault_counts:
            err(f"training_data.csv: no training rows found for fault_id '{fid}'")

# ── 4. Cross-reference: primary_symptoms in faults vs feature_ids in symptoms ──
section("4. Cross-reference: faults.json primary_symptoms vs symptoms.json feature_ids")

if "feature_ids_seen" in dir() and faults:
    for fault in faults:
        prefix = f"fault {fault.get('fault_id', '?')}"
        for sym in fault.get("primary_symptoms", []):
            expected_fid = f"symptom_{sym}"
            if expected_fid not in feature_ids_seen:
                err(
                    f"{prefix}: primary_symptom '{sym}' has no matching feature_id "
                    f"'symptom_{sym}' in symptoms.json"
                )
    print("  Cross-reference check complete.")

# ── Summary ────────────────────────────────────────────────────────────────────
section("Validation Summary")

if warnings:
    print("\n  Warnings:")
    for w in warnings:
        print(w)

if errors:
    print("\n  Errors:")
    for e in errors:
        print(e)
    print(f"\n  RESULT: FAILED -- {len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1)
else:
    print(f"\n  RESULT: PASSED -- 0 errors, {len(warnings)} warning(s)")
    print("  All knowledge base files are valid and consistent.")
    sys.exit(0)
