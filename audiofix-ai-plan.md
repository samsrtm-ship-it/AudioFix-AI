# AudioFix AI — Implementation Plan (Revised)

## Top-Level Overview

**Project Name:** AudioFix AI – Audio Equipment Troubleshooting Assistant  
**Goal:** Build a simple educational AI application that assists students, instructors, and audio technicians in diagnosing common audio equipment problems based on reported symptoms and equipment conditions.  
**Scope:** Symptom intake → ML-based diagnosis → structured troubleshooting output. No physical repair. No real-time hardware integration.  
**Approach:** A scikit-learn Decision Tree classifier (explainable, lightweight, trainable on a small curated dataset) maps binary symptom+equipment feature vectors to one of 7 core fault categories. A JSON knowledge base stores the causes, troubleshooting steps, severity, and reasoning text for each fault. A Streamlit web app provides the user interface.  
**Language:** English throughout — UI, AI outputs, and technical documentation. Standard audio terminology (microphone, mixer, gain, phantom power, feedback, distortion, hum, buzz) is used as-is.  
**Non-Goals:** Physical repair instructions, real-time audio signal analysis, hardware driver integration, LLM integration (optional future enhancement only), additional fault categories beyond the 7 core faults for the initial version.

---

## 1. Problem Definition

Audio equipment failures — microphones, mixers, speakers, cables — are common in educational and TVET settings. Diagnosing the root cause requires experience that beginners lack. Without guidance, students face prolonged downtime and learning disruption.

**Core problem the AI solves:**  
Given observable symptoms and equipment conditions reported by the user, the AI identifies the most likely fault category, lists possible causes, provides an ordered set of troubleshooting steps, assigns a severity level, and explains its reasoning — all without requiring expert knowledge.

---

## 2. Target Users

| User Group | Description |
|---|---|
| TVET / audio students | Primary users; limited troubleshooting experience |
| Audio lab technicians | Need quick, structured diagnostic checklists |
| Educators / instructors | Use the tool as a teaching aid during practical sessions |
| Hobbyists | Self-service audio troubleshooting |

---

## 3. What Problem the AI Solves

- Removes dependency on expert availability for first-level diagnosis
- Provides a consistent, repeatable troubleshooting methodology
- Educates users on why a fault occurs (transparent reasoning)
- Reduces downtime in audio labs by guiding students to the correct fix quickly
- Demonstrates applied AI/ML in a domain that TVET students understand

---

## 4. User Input (Required Information)

The user provides the following through the Streamlit form:

| Input Field | Widget Type | Options / Description |
|---|---|---|
| Equipment type | Single select (selectbox) | Microphone / Mixer / Speaker / Cable / Amplifier |
| Symptom(s) | Multi-select (multiselect) | No sound / Low volume / Distorted sound / Hum or buzz / Acoustic feedback / One channel not working / Crackling or intermittent sound |
| When did it start | Single select (selectbox) | Always / After a specific event / Gradually / Suddenly |
| Connection type | Single select (selectbox) | XLR / TRS / USB / Bluetooth / Wireless |
| Power status | Single select (selectbox) | Powered on / No power indicator / Intermittent power |
| Environment | Single select (selectbox) | Indoors / Outdoors / Live stage / Studio |
| Previous actions taken | Free text (text_area, optional) | What the user has already tried |

---

## 5. AI Output

For each diagnosis session the AI produces:

| Output Field | Description |
|---|---|
| Fault category | Named fault type (e.g. "Ground Loop Hum", "Acoustic Feedback") |
| Confidence level | High / Medium / Low — derived from classifier probability score |
| Severity level | Critical / High / Medium / Low |
| Possible causes | Ordered list of likely root causes |
| Troubleshooting steps | Numbered, actionable step-by-step instructions |
| Reasoning explanation | Plain-language explanation of why the AI reached this diagnosis (from knowledge base) |
| Escalation flag | Whether professional servicing is recommended |

---

## 6. AI / ML Approach

### Core Approach: Decision Tree Classifier (Required)

**Algorithm:** Decision Tree (primary) with Random Forest as a validation/comparison model, both via scikit-learn.

**Why Decision Tree:**
- Explainable — the decision path can be printed and shown to students
- Suitable for small, tabular, binary-feature datasets
- No GPU or internet required; fast inference
- Classification report (accuracy, precision, recall, F1) is easy to interpret for educational purposes

**Feature engineering:**
- Binary symptom features (1 = symptom present, 0 = absent): 7 features
- One-hot encoded equipment type: 5 features
- One-hot encoded connection type: 5 features
- One-hot encoded power status: 3 features
- **Total feature vector size: ~20 binary features**

**Target label:** One of 7 fault category IDs (F001–F007)

**Training data strategy:**
- Hand-authored training examples in `training_data.csv`
- Each fault category gets a minimum of 15 synthetic training examples covering typical and edge-case symptom combinations
- Total dataset: ~105–150 rows minimum
- Train/test split: 80/20

**Explainability output:**
- The classifier's decision path for the predicted class is extracted and displayed alongside the result in the UI, showing which features drove the decision

### Optional Enhancement (Not Required for Core)

| Enhancement | Description |
|---|---|
| LLM Reasoning Layer | Generate dynamic reasoning text using OpenAI API or local Ollama; falls back to static knowledge base text if disabled |

---

## 7. Dataset Structure

### 7.1 Fault Knowledge Base — `data/faults.json`

Seven records, one per core fault category. Example record:

```json
{
  "fault_id": "F001",
  "fault_name": "Microphone Not Working",
  "equipment": ["microphone"],
  "primary_symptoms": ["no_sound"],
  "secondary_symptoms": [],
  "severity": "High",
  "possible_causes": [
    "Microphone cable not connected or faulty",
    "Phantom power not enabled for condenser microphone",
    "Mute button engaged on mixer channel",
    "Gain/trim set to zero on mixer channel",
    "Faulty XLR connector"
  ],
  "troubleshooting_steps": [
    "Check that the XLR cable is firmly connected at both the microphone and mixer ends",
    "If using a condenser microphone, verify that phantom power (+48V) is enabled on the mixer channel",
    "Check that the mute button on the mixer channel is not engaged",
    "Turn up the gain/trim control on the mixer channel to an appropriate level",
    "Test with a known-working cable to rule out cable fault",
    "Test the microphone on a different channel to rule out channel fault"
  ],
  "escalate": false,
  "reasoning": "No audio from a microphone is most commonly caused by signal chain interruption — a disconnected cable, absent phantom power, or a muted/zero-gain channel — rather than internal hardware failure."
}
```

### 7.2 Symptom Vocabulary — `data/symptoms.json`

Maps UI display labels to internal feature IDs:

```json
{
  "symptom_id": "S001",
  "label": "No sound",
  "feature_id": "symptom_no_sound",
  "common_equipment": ["microphone", "speaker", "mixer", "cable"]
}
```

### 7.3 Training Dataset — `data/training_data.csv`

Binary feature matrix. Column names are the feature IDs. Target column is `fault_id`.

| symptom_no_sound | symptom_low_volume | symptom_distortion | symptom_hum_buzz | symptom_feedback | symptom_one_channel | symptom_crackling | equip_microphone | equip_mixer | equip_speaker | equip_cable | equip_amplifier | fault_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | F001 |
| 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | F005 |

### 7.4 Core Fault Categories (7 — Fixed for Initial Version)

| Fault ID | Fault Name | Maps to Symptom |
|---|---|---|
| F001 | Microphone Not Working | No sound (microphone) |
| F002 | Low Volume | Low volume |
| F003 | Distorted Sound | Distorted sound |
| F004 | Hum or Buzz | Hum or buzz |
| F005 | Acoustic Feedback | Acoustic feedback |
| F006 | One Channel Not Working | One channel not working |
| F007 | Crackling or Intermittent Sound | Crackling or intermittent sound |

> Additional fault categories (e.g. wireless dropout, RF interference, amplifier clipping) are listed as **future enhancements** and are not part of the initial version.

---

## 8. System Architecture

```
+-------------------------------------------------------+
|                  User (Web Browser)                   |
+-------------------------------------------------------+
                          |
                          | Streamlit (runs locally)
                          v
+-------------------------------------------------------+
|              Streamlit UI Layer  (app.py)             |
|  - Equipment & symptom selection widgets              |
|  - Context fields (connection type, power, env)       |
|  - "Diagnose" button                                  |
|  - Results display: fault, severity, steps, reasoning |
|  - Decision path explainability panel                 |
+-------------------------------------------------------+
                          |
                          | Python function call
                          v
+-------------------------------------------------------+
|           Engine Layer  (engine/)                     |
|                                                       |
|  normaliser.py       — raw input → feature vector     |
|  diagnoser.py        — feature vector → fault ID      |
|                        (Decision Tree classifier)     |
|  retriever.py        — fault ID → knowledge details   |
|  response_builder.py — assembles final result dict    |
+-------------------------------------------------------+
                          |
                          | File read (startup)
                          v
+-------------------------------------------------------+
|           Data Layer  (data/)                         |
|  faults.json         — fault knowledge base           |
|  symptoms.json       — symptom vocabulary             |
|  training_data.csv   — ML training examples           |
|  model.pkl           — serialised trained classifier  |
+-------------------------------------------------------+
                          |
                          | Optional (future)
                          v
+-------------------------------------------------------+
|           LLM Layer  (optional)                       |
|  OpenAI API or local Ollama endpoint                  |
|  Generates dynamic reasoning text                     |
+-------------------------------------------------------+
```

---

## 9. Project Modules

| Module | File(s) | Responsibility |
|---|---|---|
| M1 — Data Layer | `data/faults.json`, `data/symptoms.json`, `data/training_data.csv` | Stores all fault knowledge and training examples |
| M2 — Symptom Normaliser | `engine/normaliser.py` | Converts raw Streamlit form input into a binary feature vector |
| M3 — Diagnosis Engine | `engine/diagnoser.py` | Loads `model.pkl`; runs classifier on feature vector; returns fault ID, confidence score, and decision path |
| M4 — Knowledge Retrieval | `engine/retriever.py` | Looks up full fault record from `faults.json` by fault ID |
| M5 — Response Builder | `engine/response_builder.py` | Merges classifier output with knowledge base details into the final result dictionary |
| M6 — ML Trainer | `scripts/train_model.py` | Trains Decision Tree and Random Forest classifiers on `training_data.csv`; saves best model as `data/model.pkl`; prints classification report |
| M7 — Streamlit UI | `app.py` | Main Streamlit application; renders form, calls engine, displays results |
| M8 — Tests | `tests/` | Unit tests for each engine module; integration test for end-to-end diagnosis |
| M9 — Configuration | `config.py` | App settings, LLM toggle flag, confidence thresholds |
| M10 — Knowledge Base Validator | `scripts/validate_kb.py` | Checks all `faults.json` records are complete and schema-compliant |

---

## 10. Testing Strategy

| Test Type | Scope | Method |
|---|---|---|
| Knowledge base validation | All 7 fault records are complete, consistent, and schema-compliant | `scripts/validate_kb.py` run before training |
| Unit tests — normaliser | Raw input → correct binary feature vector | pytest; one test per equipment/symptom combination |
| Unit tests — diagnoser | Feature vector → correct fault ID for known inputs | pytest; one test per fault category (7 tests minimum) |
| Unit tests — retriever | Fault ID → correct knowledge record returned | pytest; verify all fields present |
| Classifier accuracy | Decision Tree performance on 20% held-out test split | scikit-learn `classification_report`; target ≥ 85% accuracy |
| Model comparison | Decision Tree vs Random Forest accuracy comparison | Printed side-by-side in training script output |
| Integration test | End-to-end: raw form input → final result dict | pytest; verify all output fields populated correctly |
| Streamlit UI test | Form renders, submit works, results display | Manual browser walkthrough checklist |
| Edge case tests | Empty symptom selection, all symptoms selected, unknown combinations | Assert graceful fallback message returned |
| Educator review | Instructors/students evaluate output quality | Structured feedback form (qualitative) |

**Accuracy Target:** ≥ 85% correct fault classification on the 20% test split.

---

## 11. Implementation Phases

### Phase 1 — Knowledge Base & Dataset Construction [x] done

**Intent:** Build and validate the fault knowledge base and training dataset that all other components depend on.  
**Expected Outcomes:** `faults.json` (7 records), `symptoms.json`, and `training_data.csv` (≥105 rows) are complete and pass schema validation.  
**Todo:**
- [ ] Finalise the 7 fault category definitions with causes, steps, severity, and reasoning text
- [ ] Author all 7 records in `data/faults.json`
- [ ] Author `data/symptoms.json` with all 7 symptom entries and feature IDs
- [ ] Author `data/training_data.csv` with ≥15 training examples per fault category
- [ ] Write `scripts/validate_kb.py` to check schema completeness
- [ ] Run validator; fix any issues until all records pass

**Relevant context:** Fault categories are fixed at F001–F007 for this version. Feature IDs in `training_data.csv` must match those defined in `symptoms.json`.

---

### Phase 2 — ML Classifier Training [x] done

**Intent:** Train, evaluate, and serialise the Decision Tree classifier that performs the core AI diagnosis.  
**Expected Outcomes:** `data/model.pkl` saved; classification report shows ≥ 85% accuracy; decision path extraction confirmed working.  
**Todo:**
- [ ] Implement `scripts/train_model.py` — load CSV, build feature matrix, train Decision Tree and Random Forest
- [ ] Print classification report (accuracy, precision, recall, F1 per class) for both models
- [ ] Save the best-performing model as `data/model.pkl` using joblib
- [ ] Confirm that `sklearn.tree.export_text` produces a readable decision path from the trained tree
- [ ] If accuracy < 85%, augment `training_data.csv` with additional examples and retrain

**Relevant context:** Feature columns in `training_data.csv` must align exactly with what `normaliser.py` produces. The Decision Tree is preferred for explainability; Random Forest is used for comparison only.

---

### Phase 3 — Engine Layer [x] done

**Intent:** Implement the Python engine modules that transform user input into a structured diagnosis result.  
**Expected Outcomes:** All four engine modules (`normaliser`, `diagnoser`, `retriever`, `response_builder`) work correctly in isolation and together.  
**Todo:**
- [ ] Implement `engine/normaliser.py` — converts raw form dict → binary feature vector (NumPy array)
- [ ] Implement `engine/diagnoser.py` — loads `model.pkl`, runs prediction, returns fault ID + confidence score + decision path text
- [ ] Implement `engine/retriever.py` — loads `faults.json`, returns full fault record by fault ID
- [ ] Implement `engine/response_builder.py` — merges diagnoser output + retriever output into final result dict
- [ ] Write unit tests in `tests/` for each module
- [ ] Run all unit tests; confirm they pass

**Relevant context:** `diagnoser.py` must use `sklearn.tree.export_text` on the trained tree to extract the decision path for the predicted class. The feature vector order in `normaliser.py` must exactly match the column order in `training_data.csv`.

---

### Phase 4 — Streamlit UI [x] done

**Intent:** Build the user-facing Streamlit application that collects symptoms and displays the diagnosis result in a clean, student-friendly layout.  
**Expected Outcomes:** A locally runnable Streamlit app where a user can select symptoms, click Diagnose, and see a fully structured result with fault name, severity badge, troubleshooting steps, reasoning, and AI decision path.  
**Todo:**
- [ ] Implement `app.py` with Streamlit input widgets for all required fields
- [ ] Wire "Diagnose" button to call the engine layer and retrieve results
- [ ] Display results: fault category, confidence level (colour-coded), severity badge, possible causes, troubleshooting steps (numbered), reasoning text
- [ ] Display AI explainability panel: Decision Tree decision path shown in a collapsible section
- [ ] Display escalation warning if `escalate: true` in the fault record
- [ ] Handle the edge case where no symptoms are selected (show a friendly prompt)
- [ ] Test the full flow manually in the browser for all 7 fault categories

**Relevant context:** No Flask or REST API is needed — Streamlit calls engine Python functions directly in the same process. The decision path panel is a key educational feature demonstrating how the AI reached its conclusion.

---

### Phase 5 — Integration Testing & Validation [ ] pending

**Intent:** Confirm the complete system meets accuracy targets, handles edge cases gracefully, and is ready for educational use.  
**Expected Outcomes:** All tests pass; classification report archived; UI checklist completed; README written.  
**Todo:**
- [ ] Run full pytest suite (unit + integration tests)
- [ ] Archive the final classification report as `docs/classification_report.txt`
- [ ] Complete the manual Streamlit UI browser checklist for all 7 fault categories
- [ ] Test edge cases: no symptoms selected, all symptoms selected, single symptom only
- [ ] Conduct a review session with 1–2 audio students or instructors; collect feedback
- [ ] Fix any critical issues identified
- [ ] Write `README.md` with project description, setup instructions, and how to run the app

---

### Phase 6 — LLM Reasoning Layer (Optional / Future Enhancement) [ ] pending

**Intent:** Optionally replace static reasoning text with dynamically generated natural-language explanations from an LLM.  
**Expected Outcomes:** When enabled via config flag, the `reasoning` field is generated by an LLM using the fault details as context; falls back to static `faults.json` text when disabled.  
**Todo:**
- [ ] Add `LLM_ENABLED` toggle to `config.py`
- [ ] Implement LLM client in `engine/response_builder.py` (OpenAI API or local Ollama)
- [ ] Design prompt template using fault name, causes, and steps as context
- [ ] Test generated reasoning quality vs static text
- [ ] Ensure graceful fallback when LLM is disabled or unavailable

> **Note:** This phase is not required for the core project. It should only be attempted after Phase 5 is fully complete and validated.

---

## File & Folder Structure

```
AudioFix-AI/
│
├── app.py                        # Streamlit main application
├── config.py                     # App settings and feature flags
├── requirements.txt              # Python dependencies
├── README.md                     # Setup and usage guide
│
├── data/
│   ├── faults.json               # Fault knowledge base (7 records)
│   ├── symptoms.json             # Symptom vocabulary
│   ├── training_data.csv         # ML training examples
│   └── model.pkl                 # Serialised trained classifier
│
├── engine/
│   ├── __init__.py
│   ├── normaliser.py             # Input → feature vector
│   ├── diagnoser.py              # Feature vector → fault ID + confidence
│   ├── retriever.py              # Fault ID → knowledge record
│   └── response_builder.py       # Assemble final result
│
├── scripts/
│   ├── train_model.py            # Train and save classifier
│   └── validate_kb.py            # Validate faults.json schema
│
├── tests/
│   ├── test_normaliser.py
│   ├── test_diagnoser.py
│   ├── test_retriever.py
│   ├── test_response_builder.py
│   └── test_integration.py
│
└── docs/
    └── classification_report.txt # Archived model accuracy report
```

---

## Python Dependencies

```
streamlit
scikit-learn
pandas
numpy
joblib
pytest
```

---

## Status Legend

- `[ ] pending` — not started
- `[-] in progress` — actively being worked
- `[x] done` — completed and verified
