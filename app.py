"""
app.py
------
AudioFix AI – Audio Equipment Troubleshooting Assistant
Streamlit web application.

Run with:
    streamlit run app.py
"""

import streamlit as st

from engine.normaliser import get_equipment_labels, get_symptom_labels
from engine.response_builder import build_response

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AudioFix AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _severity_badge(severity: str) -> str:
    colours = {
        "Critical": ("#fff0f0", "#d62728", "⛔"),
        "High":     ("#fff4e6", "#e65c00", "🔴"),
        "Medium":   ("#fffbe6", "#b8860b", "🟡"),
        "Low":      ("#f0fff4", "#276221", "🟢"),
    }
    bg, fg, icon = colours.get(severity, ("#f5f5f5", "#333", "⚪"))
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg};'
        f'border-radius:4px;padding:2px 10px;font-weight:600;font-size:0.9rem;">'
        f'{icon} {severity}</span>'
    )


def _confidence_badge(label: str, score: float) -> str:
    colours = {
        "High":   ("#f0fff4", "#276221"),
        "Medium": ("#fffbe6", "#b8860b"),
        "Low":    ("#fff0f0", "#d62728"),
    }
    bg, fg = colours.get(label, ("#f5f5f5", "#333"))
    pct = f"{score * 100:.0f}%"
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg};'
        f'border-radius:4px;padding:2px 10px;font-weight:600;font-size:0.9rem;">'
        f'{label} ({pct})</span>'
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="background:#1a1a2e;padding:22px 28px 20px;border-radius:8px;margin-bottom:8px;">
        <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;">
            <span style="color:#e0e0ff;font-size:2.1rem;font-weight:800;
                         letter-spacing:2px;line-height:1.1;">
                🎙️ AUDIO FIX AI
            </span>
            <span style="color:#c8b8ff;font-size:1.15rem;
                         font-family:'Segoe Script','Brush Script MT','Comic Sans MS',
                         'Dancing Script',cursive;
                         font-style:italic;font-weight:400;line-height:1.1;">
                by Sams Audio Garage
            </span>
        </div>
        <p style="color:#a0a0cc;margin:8px 0 0;font-size:1.0rem;
                  text-align:left;letter-spacing:0.5px;">
            Audio Equipment Troubleshooting Assistant
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "_Describe the problem with your audio equipment and let AI help identify "
    "the most likely cause and recommended troubleshooting steps._"
)
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("📋 Step 1 — Describe the Problem")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    equipment = st.selectbox(
        "Equipment type *",
        options=get_equipment_labels(),
        index=0,
        help="Select the piece of audio equipment that is experiencing the problem.",
    )

    symptoms = st.multiselect(
        "Symptom(s) observed *",
        options=get_symptom_labels(),
        default=[],
        help="Select all symptoms that apply. At least one symptom is recommended for an accurate diagnosis.",
    )

    when_started = st.selectbox(
        "When did the problem start?",
        options=["Always", "After a specific event", "Gradually", "Suddenly"],
        help="This helps narrow down whether the fault is new or long-standing.",
    )

with col_right:
    connection_type = st.selectbox(
        "Connection type",
        options=["XLR", "TRS / TS (jack)", "USB", "Bluetooth", "Wireless (RF)"],
        help="The type of cable or connection being used.",
    )

    power_status = st.selectbox(
        "Power status",
        options=["Powered on (normal)", "No power indicator", "Intermittent power"],
        help="The current power state of the equipment.",
    )

    environment = st.selectbox(
        "Environment",
        options=["Indoors – classroom / studio", "Indoors – live venue", "Outdoors", "Live stage"],
        help="The environment where the equipment is being used.",
    )

    notes = st.text_area(
        "Additional notes (optional)",
        placeholder="e.g. Already tried replacing the cable. Problem started after a live show.",
        height=88,
        help="Describe anything else you have already tried or noticed.",
    )

st.divider()

# ---------------------------------------------------------------------------
# Diagnose button
# ---------------------------------------------------------------------------
diagnose_clicked = st.button(
    "🔍 Diagnose",
    type="primary",
    use_container_width=False,
)

# ---------------------------------------------------------------------------
# Diagnosis logic
# ---------------------------------------------------------------------------
if diagnose_clicked:
    # Guard: warn if no symptom is selected
    if not symptoms:
        st.warning(
            "⚠️ No symptom selected. Please select at least one symptom from the list "
            "for an accurate diagnosis. The AI will still attempt a diagnosis, but "
            "confidence will be low.",
            icon="⚠️",
        )

    # Build the user_input dict and run the engine
    user_input = {
        "symptoms":        symptoms,
        "equipment":       [equipment],
        "when_started":    when_started,
        "connection_type": connection_type,
        "power_status":    power_status,
        "environment":     environment,
        "notes":           notes,
    }

    with st.spinner("Analysing symptoms…"):
        result = build_response(user_input)

    # ── Results header ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🩺 Step 2 — Diagnosis Result")

    # Fault name + badges row
    badges_html = (
        _severity_badge(result["severity"])
        + "&nbsp;&nbsp;"
        + _confidence_badge(result["confidence_label"], result["confidence"])
    )
    st.markdown(
        f"<h3 style='margin-bottom:4px;'>{result['fault_name']}</h3>"
        f"<div style='margin-bottom:12px;'>{badges_html}</div>",
        unsafe_allow_html=True,
    )

    # Escalation warning
    if result["escalate"]:
        st.error(
            "🔧 **Professional Service Recommended** — Based on the diagnosis, "
            "this fault may require inspection or repair by a qualified audio technician. "
            "Do not attempt to open or modify internal components.",
            icon="🔧",
        )

    # ── Main result columns ───────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("#### 🔎 Possible Causes")
        for i, cause in enumerate(result["possible_causes"], start=1):
            st.markdown(f"**{i}.** {cause}")

    with col_b:
        st.markdown("#### 🛠️ Troubleshooting Steps")
        for i, step in enumerate(result["troubleshooting_steps"], start=1):
            st.markdown(f"**Step {i}:** {step}")

    st.divider()

    # ── AI Reasoning ─────────────────────────────────────────────────────
    st.markdown("#### 💡 AI Reasoning")
    st.info(result["reasoning"], icon="💡")

    # ── Probability breakdown ─────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 AI Confidence by Fault Category")

    # Import fault names from retriever for the chart labels
    from engine.retriever import get_fault

    prob_data = sorted(
        result["all_probabilities"].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    fault_labels = []
    fault_probs = []
    for fid, prob in prob_data:
        try:
            name = get_fault(fid)["fault_name"]
        except KeyError:
            name = fid
        fault_labels.append(f"{fid} – {name}")
        fault_probs.append(round(prob * 100, 1))

    # Streamlit bar chart using st.bar_chart with a DataFrame
    import pandas as pd
    chart_df = pd.DataFrame(
        {"Confidence (%)": fault_probs},
        index=fault_labels,
    )
    st.bar_chart(chart_df, height=260, use_container_width=True)

    # ── AI Explainability — Decision Tree path ────────────────────────────
    st.divider()
    with st.expander("🌳 AI Explainability — How the Decision Tree reached this diagnosis", expanded=False):
        st.markdown(
            """
            The diagnosis above was made by a **Decision Tree classifier** trained on
            labelled audio equipment fault examples.

            The tree below shows the complete set of rules the AI used.
            Each branch tests whether a symptom was present (> 0.50 means **Yes**,
            ≤ 0.50 means **No**). The tree follows the branches that match your
            selected symptoms until it reaches a **class** — the predicted fault.

            **Your selected symptoms / active features:**
            """
        )

        if result["active_features"]:
            feature_list = ", ".join(
                f"`{f}`" for f in result["active_features"]
            )
            st.markdown(f"&nbsp;&nbsp;{feature_list}")
        else:
            st.markdown("&nbsp;&nbsp;_No symptoms selected — all features were 0._")

        st.markdown("---")
        st.markdown("**Full Decision Tree:**")
        st.code(result["decision_path"], language="text")

        st.markdown(
            """
            ---
            > **Educational note:** A Decision Tree makes decisions by asking a series
            > of yes/no questions about the input features (symptoms and equipment type).
            > Each question narrows down the possible fault until only one remains.
            > This makes the AI's reasoning fully transparent — you can trace exactly
            > which symptoms led to the diagnosis.
            """
        )

    # ── Session summary card ──────────────────────────────────────────────
    st.divider()
    with st.expander("📋 Session Summary", expanded=False):
        st.markdown("**Inputs provided:**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"- **Equipment:** {equipment}")
            st.markdown(
                f"- **Symptom(s):** {', '.join(symptoms) if symptoms else '_None selected_'}"
            )
            st.markdown(f"- **When started:** {when_started}")
        with c2:
            st.markdown(f"- **Connection type:** {connection_type}")
            st.markdown(f"- **Power status:** {power_status}")
            st.markdown(f"- **Environment:** {environment}")
        if notes.strip():
            st.markdown(f"- **Notes:** {notes.strip()}")

        st.markdown("**Diagnosis:**")
        st.markdown(f"- **Fault:** {result['fault_name']} ({result['fault_id']})")
        st.markdown(f"- **Severity:** {result['severity']}")
        st.markdown(
            f"- **Confidence:** {result['confidence_label']} "
            f"({result['confidence'] * 100:.0f}%)"
        )
        st.markdown(
            f"- **Professional service recommended:** "
            f"{'Yes' if result['escalate'] else 'No'}"
        )

# ---------------------------------------------------------------------------
# Footer (shown always)
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align:center;padding:10px 0 6px;">
        <p style="color:#a0a0cc;font-size:0.88rem;margin:0;font-weight:600;
                  letter-spacing:1px;">
            &copy; 2026 SAMS AUDIO GARAGE
        </p>
        <p style="color:#666;font-size:0.78rem;margin:4px 0 0;">
            AudioFix AI &mdash; Audio Equipment Troubleshooting Assistant
        </p>
        <p style="color:#555;font-size:0.74rem;margin:4px 0 0;">
            This tool provides preliminary diagnosis guidance only and does not
            replace a qualified audio technician.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
