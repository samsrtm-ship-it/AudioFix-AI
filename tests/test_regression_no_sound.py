"""
tests/test_regression_no_sound.py
----------------------------------
Regression tests for the equipment-specific "No sound" prediction bug.

Bug reported: Speaker + No sound was incorrectly diagnosed as F001
(Microphone Not Working) because the trained model did not use equipment
features to distinguish between microphone no-sound and speaker/amplifier
no-sound.

Fix: Added F008 (No Audio Output - Speaker / Amplifier / Mixer) to the
knowledge base and training dataset, and cleaned F001 training rows to be
strictly microphone-only. Model was retrained.

These tests must continue to pass on every future model retrain.
"""

import pytest

from engine.response_builder import build_response

# ---------------------------------------------------------------------------
# Required correct fault mappings
# ---------------------------------------------------------------------------

# F001: Microphone Not Working  — only when microphone is selected
# F008: No Audio Output – Speaker / Amplifier / Mixer — all non-mic equipment

F001 = "F001"
F008 = "F008"


def _diagnose(symptoms, equipment):
    return build_response({"symptoms": symptoms, "equipment": [equipment]})


# ---------------------------------------------------------------------------
# Core regression — the four scenarios from the bug report
# ---------------------------------------------------------------------------

class TestNoSoundEquipmentRegression:

    def test_microphone_no_sound_is_F001(self):
        """Microphone + No sound MUST be F001 (Microphone Not Working)."""
        result = _diagnose(["No sound"], "Microphone")
        assert result["fault_id"] == F001, (
            f"Regression: Microphone + No sound predicted {result['fault_id']} "
            f"({result['fault_name']}), expected {F001}"
        )

    def test_speaker_no_sound_is_NOT_F001(self):
        """
        Speaker + No sound must NOT be F001 (Microphone Not Working).
        This is the exact bug that was reported in the live application.
        """
        result = _diagnose(["No sound"], "Speaker")
        assert result["fault_id"] != F001, (
            f"REGRESSION BUG: Speaker + No sound is predicting {F001} "
            f"(Microphone Not Working). This is incorrect."
        )

    def test_speaker_no_sound_is_F008(self):
        """Speaker + No sound MUST be F008 (No Audio Output – Speaker/Amplifier/Mixer)."""
        result = _diagnose(["No sound"], "Speaker")
        assert result["fault_id"] == F008, (
            f"Speaker + No sound predicted {result['fault_id']} "
            f"({result['fault_name']}), expected {F008}"
        )

    def test_amplifier_no_sound_is_NOT_F001(self):
        """Amplifier + No sound must NOT be F001 (Microphone Not Working)."""
        result = _diagnose(["No sound"], "Amplifier")
        assert result["fault_id"] != F001, (
            f"Regression: Amplifier + No sound predicted F001 (Microphone Not Working)."
        )

    def test_amplifier_no_sound_is_F008(self):
        """Amplifier + No sound MUST be F008."""
        result = _diagnose(["No sound"], "Amplifier")
        assert result["fault_id"] == F008, (
            f"Amplifier + No sound predicted {result['fault_id']}, expected {F008}"
        )

    def test_mixer_no_sound_is_NOT_F001(self):
        """Mixer + No sound must NOT be F001 (Microphone Not Working)."""
        result = _diagnose(["No sound"], "Mixer")
        assert result["fault_id"] != F001, (
            f"Regression: Mixer + No sound predicted F001 (Microphone Not Working)."
        )

    def test_mixer_no_sound_is_F008(self):
        """Mixer + No sound MUST be F008."""
        result = _diagnose(["No sound"], "Mixer")
        assert result["fault_id"] == F008, (
            f"Mixer + No sound predicted {result['fault_id']}, expected {F008}"
        )


# ---------------------------------------------------------------------------
# F008 result structure — confirm KB content is populated
# ---------------------------------------------------------------------------

class TestF008KnowledgeBaseContent:

    def test_f008_fault_name_correct(self):
        result = _diagnose(["No sound"], "Speaker")
        assert "Speaker" in result["fault_name"] or "Amplifier" in result["fault_name"], (
            f"F008 fault_name does not mention Speaker or Amplifier: {result['fault_name']}"
        )

    def test_f008_has_possible_causes(self):
        result = _diagnose(["No sound"], "Speaker")
        assert len(result["possible_causes"]) >= 3

    def test_f008_has_troubleshooting_steps(self):
        result = _diagnose(["No sound"], "Speaker")
        assert len(result["troubleshooting_steps"]) >= 3

    def test_f008_has_reasoning(self):
        result = _diagnose(["No sound"], "Speaker")
        assert len(result["reasoning"].strip()) > 0

    def test_f008_escalate_is_bool(self):
        result = _diagnose(["No sound"], "Speaker")
        assert isinstance(result["escalate"], bool)

    def test_f008_severity_is_valid(self):
        result = _diagnose(["No sound"], "Speaker")
        assert result["severity"] in ("Critical", "High", "Medium", "Low")

    def test_f008_high_confidence(self):
        """Clean single-symptom input for F008 should return High confidence."""
        result = _diagnose(["No sound"], "Speaker")
        assert result["confidence_label"] == "High"
        assert result["confidence"] >= 0.80


# ---------------------------------------------------------------------------
# Cross-check: other symptoms on speaker still work correctly
# ---------------------------------------------------------------------------

class TestSpeakerOtherSymptoms:

    def test_speaker_low_volume_is_F002(self):
        result = _diagnose(["Low volume"], "Speaker")
        assert result["fault_id"] == "F002"

    def test_speaker_distortion_is_F003(self):
        result = _diagnose(["Distorted sound"], "Speaker")
        assert result["fault_id"] == "F003"

    def test_speaker_hum_is_F004(self):
        result = _diagnose(["Hum or buzz"], "Speaker")
        assert result["fault_id"] == "F004"

    def test_speaker_crackling_is_F007(self):
        result = _diagnose(["Crackling or intermittent sound"], "Speaker")
        assert result["fault_id"] == "F007"


# ---------------------------------------------------------------------------
# Cross-check: microphone other symptoms still work
# ---------------------------------------------------------------------------

class TestMicrophoneOtherSymptoms:

    def test_microphone_no_sound_still_F001(self):
        """Ensure F001 was not broken by the fix."""
        result = _diagnose(["No sound"], "Microphone")
        assert result["fault_id"] == F001

    def test_microphone_distortion_is_F003(self):
        result = _diagnose(["Distorted sound"], "Microphone")
        assert result["fault_id"] == "F003"

    def test_microphone_feedback_is_F005(self):
        result = _diagnose(["Acoustic feedback"], "Microphone")
        assert result["fault_id"] == "F005"
