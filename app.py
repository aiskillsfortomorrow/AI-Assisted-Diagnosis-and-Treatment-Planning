import os
from datetime import datetime

import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CareCompass | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
)

MODELS = {
    "Llama 3.3 70B (most capable, recommended)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (faster, simpler answers)": "llama-3.1-8b-instant",
}

# ---------------------------------------------------------------------------
# Pastel styling
# Works in BOTH light and dark browser modes: every text colour, input colour
# and button colour is forced, so labels can never turn white-on-white.
# ---------------------------------------------------------------------------
PASTEL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&family=Quicksand:wght@600;700&display=swap');

:root {
    --mint: #D8F3E6;  --lavender: #E6DFF7;  --peach: #FFE3D3;
    --sky: #D9EAFB;   --butter: #FFF4CC;    --rose: #FADADD;
    --ink: #3B3A52;   --ink-soft: #66657F;
    --accent: #7C67C0; --accent-dark: #6552A8;
}

html, body, .stApp { font-family: 'Nunito', sans-serif; }
.stApp { background: linear-gradient(160deg, #FBF8FF 0%, #EEF7FF 50%, #F2FBF6 100%); }
#MainMenu, footer { visibility: hidden; }

/* ---- Force readable dark text on every pastel surface ---- */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th,
[data-testid="stMarkdownContainer"] *, [data-testid="stWidgetLabel"] *,
[data-testid="stCaptionContainer"] *, [data-testid="stAlert"] *,
[data-testid="stExpander"] summary *, [data-testid="stChatMessage"] * {
    color: var(--ink);
}
[data-testid="stCaptionContainer"] * { color: var(--ink-soft) !important; }
h1, h2, h3, h4 { font-family: 'Quicksand', sans-serif !important; }

/* ---- Inputs: always white with dark text ---- */
input, textarea,
div[data-baseweb="input"], div[data-baseweb="base-input"],
div[data-baseweb="textarea"], div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    border-radius: 14px !important;
}
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div {
    border: 1.5px solid #CFC6EE !important;
}
div[data-baseweb="select"] * { color: var(--ink) !important; }
div[data-baseweb="select"] svg { fill: var(--ink) !important; }
::placeholder { color: #9593AE !important; -webkit-text-fill-color: #9593AE !important; opacity: 1 !important; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124, 103, 192, 0.22) !important;
}

/* Number box +/- buttons */
[data-testid="stNumberInput"] button {
    background: var(--lavender) !important; border: none !important;
}
[data-testid="stNumberInput"] button svg { fill: var(--ink) !important; color: var(--ink) !important; }

/* Dropdown list */
ul[role="listbox"], div[data-baseweb="popover"] ul, div[data-baseweb="menu"] {
    background: #FFFFFF !important;
}
ul[role="listbox"] li, div[data-baseweb="menu"] li, div[data-baseweb="menu"] * {
    color: var(--ink) !important; background: #FFFFFF !important;
}
ul[role="listbox"] li:hover { background: var(--lavender) !important; }

/* ---- Hero & notices ---- */
.hero {
    background: linear-gradient(120deg, var(--lavender), var(--sky) 55%, var(--mint));
    border-radius: 28px; padding: 1.8rem 2.2rem; margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(124, 103, 192, 0.15);
}
.hero h1 { margin: 0 0 .3rem 0; font-size: 2.1rem; }
.hero p  { margin: 0; font-size: 1.05rem; color: var(--ink-soft) !important; }
.notice {
    background: var(--butter); border-radius: 18px; padding: .8rem 1.1rem;
    margin-bottom: 1rem; font-size: .95rem; border-left: 6px solid #E9CC5E;
}

/* ---- Step cards ---- */
.step-card { border-radius: 20px; padding: 1rem 1.2rem; box-shadow: 0 4px 14px rgba(120,110,170,.10); min-height: 118px; }
.step-card h4 { margin: 0 0 .3rem 0; font-size: 1.05rem; }
.step-card p  { margin: 0; font-size: .93rem; color: var(--ink-soft) !important; }
.step-1 { background: var(--lavender); } .step-2 { background: var(--peach); } .step-3 { background: var(--mint); }

/* ---- Form section banners ---- */
.section-banner { background: var(--sky); border-radius: 16px; padding: .7rem 1rem; margin: 1.4rem 0 .8rem 0; }
.section-banner b { font-family: 'Quicksand', sans-serif; font-size: 1.1rem; }
.section-banner span { display: block; font-size: .92rem; color: var(--ink-soft) !important; }

/* ---- Field labels (our own, always visible) ---- */
.fl { margin: .35rem 0 .25rem 0; line-height: 1.3; }
.fl-title { font-weight: 800; font-size: 1rem; color: #3B3A52 !important; }
.fl-title .req { color: #C2415D !important; font-weight: 800; font-size: .85rem; }
.fl-title .opt { color: #7C7A96 !important; font-weight: 600; font-size: .85rem; }
.fl-hint { font-size: .85rem; color: #66657F !important; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background: var(--lavender); }

/* ---- Tabs ---- */
button[data-baseweb="tab"] { border-radius: 14px 14px 0 0; }
button[data-baseweb="tab"] * { font-family: 'Quicksand', sans-serif; font-weight: 700; font-size: 1rem; color: var(--ink) !important; }
button[data-baseweb="tab"][aria-selected="true"] { background: var(--mint); }

/* ---- Buttons: white text on purple ---- */
.stApp .stButton > button, .stApp .stDownloadButton > button, .stApp .stFormSubmitButton > button {
    background: var(--accent) !important; border: none !important; border-radius: 999px !important;
    padding: .6rem 1.6rem !important; transition: transform .15s ease, background .15s ease;
}
.stApp .stButton > button *, .stApp .stDownloadButton > button *, .stApp .stFormSubmitButton > button * {
    color: #FFFFFF !important; font-family: 'Quicksand', sans-serif; font-weight: 700;
}
.stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover,
.stApp .stFormSubmitButton > button:hover { background: var(--accent-dark) !important; transform: translateY(-1px); }

/* ---- Containers ---- */
[data-testid="stExpander"], [data-testid="stChatMessage"], [data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF; border-radius: 18px;
}
[data-testid="stForm"] { background: rgba(255,255,255,.55); border: 1.5px solid #DDD5F3; border-radius: 24px; padding: 1.2rem 1.4rem; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(PASTEL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data and helpers
# ---------------------------------------------------------------------------
SEX_OPTIONS = ["Not stated", "Female", "Male", "Other"]

ANALYSIS_TYPES = {
    "🔍 Health risk assessment": {
        "help": "Lists the main health risks (low / moderate / high), what drives each one, and what to monitor. Red flags come first.",
        "task": (
            "Identify the patient's main health risks (for example deterioration, sepsis, "
            "kidney injury, cardiovascular events, falls, readmission). For each risk give: "
            "a risk level (low/moderate/high), the findings that drive it, and monitoring "
            "or preventive steps. List any red-flag findings first."
        ),
    },
    "🧭 Differential diagnosis": {
        "help": "A ranked list of possible diagnoses, with evidence for and against each, and the next test that would help most.",
        "task": (
            "Provide a ranked differential diagnosis. For each possibility give the "
            "supporting findings, the findings against it, and the next test that would "
            "best confirm or exclude it. Highlight any 'cannot miss' diagnoses."
        ),
    },
    "💊 Treatment support plan": {
        "help": "Treatment options, medication safety checks (interactions, allergies, kidney/liver dosing), monitoring and follow-up.",
        "task": (
            "Suggest a treatment support strategy: guideline-based options, medication "
            "safety checks (interactions, allergies, renal/hepatic dosing, contraindications), "
            "monitoring plan, follow-up, and when to escalate. Do not give exact "
            "prescriptions; describe options for clinician decision."
        ),
    },
    "📋 Complete care summary": {
        "help": "One combined report: key findings, risks, possible diagnoses and treatment support.",
        "task": (
            "Produce a complete summary with four sections: 1) Key findings, 2) Health risks, "
            "3) Possible diagnoses, 4) Treatment support and follow-up. Keep it concise."
        ),
    },
    "💬 Plain-language patient explanation": {
        "help": "A simple, friendly explanation to share with the patient or family, including warning signs to watch for.",
        "task": (
            "Write a warm, simple explanation for the patient at a 6th-grade reading level: "
            "what may be going on, what happens next, and warning signs that need urgent "
            "care. Avoid jargon and avoid alarming language."
        ),
    },
}

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant helping hospital clinicians. "
    "You never replace clinical judgment. Be accurate, concise, and structured. "
    "State uncertainty clearly, mention red-flag findings that need urgent attention, "
    "and recommend further tests or specialist input when appropriate. "
    "Do not invent patient data. If information is missing, say what is needed. "
    "Always end with a one-line reminder that a clinician must verify the output."
)

EXAMPLE_PATIENT = {
    "age": 68, "sex": "Male", "weight": 82.0,
    "complaint": "3 days of fever, productive cough and shortness of breath. "
                 "Feels very tired and has passed less urine than usual.",
    "hr": 104, "bp": "104/66", "temp_c": 38.6, "spo2": 92, "rr": 22,
    "history": "Type 2 diabetes, high blood pressure, chronic kidney disease stage 3",
    "meds": "Metformin 500 mg twice daily, amlodipine 5 mg once daily, ibuprofen as needed",
    "allergies": "Penicillin (rash)",
    "labs": "WBC 16.2, creatinine 1.9 (usual 1.4), lactate 2.8. Chest X-ray: right lower lobe opacity",
    "extra": "",
}
EMPTY_PATIENT = {
    "age": None, "sex": "Not stated", "weight": None, "complaint": "",
    "hr": None, "bp": "", "temp_c": None, "spo2": None, "rr": None,
    "history": "", "meds": "", "allergies": "", "labs": "", "extra": "",
}


def fill_example():
    for k, v in EXAMPLE_PATIENT.items():
        st.session_state[k] = v


def clear_form():
    for k, v in EMPTY_PATIENT.items():
        st.session_state[k] = v


def field(title: str, hint: str = "", required: bool = False):
    """A visible label placed above a widget: title, required/optional tag, and a hint line."""
    tag = '<span class="req"> * required</span>' if required else '<span class="opt"> (optional)</span>'
    hint_html = f'<div class="fl-hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="fl"><div class="fl-title">{title}{tag}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, hint: str):
    st.markdown(
        f'<div class="section-banner"><b>{title}</b><span>{hint}</span></div>',
        unsafe_allow_html=True,
    )


def get_secret_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def stream_response(client, messages, model, temp):
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=temp, max_tokens=1800, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def build_patient_summary(data: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in data.items() if v not in ("", None))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Optional settings")
    st.caption("You can ignore this panel. The defaults work well.")

    field("AI model", "Recommended: the first option.")
    model_label = st.selectbox("AI model", list(MODELS.keys()), label_visibility="collapsed")
    model_id = MODELS[model_label]

    field("Answer style", "Left = careful and consistent. Right = more varied. Recommended: 0.2")
    temperature = st.slider("Answer style", 0.0, 1.0, 0.2, 0.05, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🔒 Privacy reminder")
    st.markdown(
        "Do **not** type real patient names, ID numbers, phone numbers or addresses. "
        "Use de-identified or sample data only."
    )

# ---------------------------------------------------------------------------
# Header + how-to
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🩺 CareCompass</h1>
        <p>AI-assisted support for risk spotting, diagnosis ideas, and treatment planning.</p>
    </div>
    <div class="notice">
        <b>Decision support only.</b> This tool does not diagnose or prescribe.
        A qualified clinician must review every output before it is used in patient care.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 👋 How to use this app")
c1, c2, c3 = st.columns(3)
c1.markdown('<div class="step-card step-1"><h4>Step 1: Connect</h4><p>Paste your Groq API key in the box just below (only if it asks for it).</p></div>', unsafe_allow_html=True)
c2.markdown('<div class="step-card step-2"><h4>Step 2: Fill in the form</h4><p>Open the <b>Patient analysis</b> tab. Each box has a title and an example. Only age and symptoms are required.</p></div>', unsafe_allow_html=True)
c3.markdown('<div class="step-card step-3"><h4>Step 3: Get results</h4><p>Pick what you need in section E, press <b>Analyze patient</b>, then read or download the report.</p></div>', unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------------------
# Step 1: API key
# ---------------------------------------------------------------------------
st.markdown("### 🔑 Step 1: Connect to Groq")
api_key = get_secret_key()
if api_key:
    st.success("Connected. Your API key was found in the app's secure settings.", icon="✅")
else:
    field("Groq API key", "Paste the key that starts with gsk_ . Get a free one at console.groq.com → API Keys.", required=True)
    api_key = st.text_input("Groq API key", type="password", placeholder="gsk_...", label_visibility="collapsed")
    if api_key:
        st.success("Key entered. Continue to Step 2.", icon="✅")
    else:
        st.info("No key yet. Paste your Groq API key above to turn the app on.", icon="👆")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_analysis, tab_chat, tab_about = st.tabs(["🌷 Patient analysis", "💬 Ask a question", "ℹ️ About & help"])

# =========================== TAB 1: Patient analysis ========================
with tab_analysis:
    st.markdown("### 📝 Step 2: Enter patient details")
    st.caption("Fill in what you know. Only **age** and **symptoms** are required. More detail gives a better analysis.")

    b1, b2, _ = st.columns([1.4, 1, 3])
    b1.button("🧪 Fill with example patient", on_click=fill_example)
    b2.button("🧹 Clear form", on_click=clear_form)
    st.caption("Tip: press **Fill with example patient** to see what a completed form looks like (fictional data).")

    with st.form("patient_form"):
        # ---- A ----
        section("A. Patient basics", "Who is the patient?")
        a1, a2, a3 = st.columns(3)
        with a1:
            field("Age (years)", "Whole number, e.g. 68", required=True)
            st.number_input("Age in years", 0, 120, value=None, step=1, key="age",
                            placeholder="Type age here", label_visibility="collapsed")
        with a2:
            field("Sex", "Choose from the list")
            st.selectbox("Sex", SEX_OPTIONS, key="sex", label_visibility="collapsed")
        with a3:
            field("Weight (kg)", "Helps with medicine doses, e.g. 70")
            st.number_input("Weight in kg", 0.0, 300.0, value=None, step=0.5, key="weight",
                            placeholder="Type weight here", label_visibility="collapsed")

        # ---- B ----
        section("B. Symptoms and complaint", "Why has the patient come in?")
        field("Main complaint and symptoms",
              "What is wrong, for how long, and what makes it better or worse. "
              "Example: 3 days of fever, cough and shortness of breath.", required=True)
        st.text_area("Main complaint and symptoms", key="complaint", height=110,
                     placeholder="Describe the symptoms here", label_visibility="collapsed")

        # ---- C ----
        section("C. Vital signs", "Current body measurements. Leave any box empty if you do not have it.")
        r1a, r1b, r1c = st.columns(3)
        with r1a:
            field("Heart rate (beats per minute)", "Normal adult: about 60 to 100")
            st.number_input("Heart rate", 0, 250, value=None, key="hr",
                            placeholder="e.g. 80", label_visibility="collapsed")
        with r1b:
            field("Blood pressure (mmHg)", "Top number / bottom number")
            st.text_input("Blood pressure", key="bp", placeholder="e.g. 120/80", label_visibility="collapsed")
        with r1c:
            field("Body temperature (°C)", "Normal: about 36.5 to 37.5")
            st.number_input("Temperature", 30.0, 45.0, value=None, step=0.1, key="temp_c",
                            placeholder="e.g. 37.0", label_visibility="collapsed")
        r2a, r2b, _r2c = st.columns(3)
        with r2a:
            field("Oxygen level, SpO₂ (%)", "From the finger clip. Normal: 95 to 100")
            st.number_input("Oxygen level", 0, 100, value=None, key="spo2",
                            placeholder="e.g. 97", label_visibility="collapsed")
        with r2b:
            field("Breathing rate (breaths per minute)", "Normal adult: about 12 to 20")
            st.number_input("Breathing rate", 0, 60, value=None, key="rr",
                            placeholder="e.g. 16", label_visibility="collapsed")

        # ---- D ----
        section("D. History, medicines and allergies", "Background that affects safe treatment.")
        d1, d2 = st.columns(2)
        with d1:
            field("Past medical conditions", "Long-term illnesses, past surgeries")
            st.text_area("Past medical conditions", key="history", height=90,
                         placeholder="e.g. diabetes, high blood pressure, kidney disease",
                         label_visibility="collapsed")
        with d2:
            field("Current medicines", "Name, dose and how often. Include over-the-counter medicines")
            st.text_area("Current medicines", key="meds", height=90,
                         placeholder="e.g. metformin 500 mg twice daily", label_visibility="collapsed")
        d3, d4 = st.columns(2)
        with d3:
            field("Allergies", "Write 'none known' if there are none")
            st.text_input("Allergies", key="allergies", placeholder="e.g. penicillin (rash)",
                          label_visibility="collapsed")
        with d4:
            field("Test results and scans", "Blood tests, X-ray, ECG and so on")
            st.text_input("Test results", key="labs",
                          placeholder="e.g. WBC 16, creatinine 1.8, chest X-ray: right lower lobe opacity",
                          label_visibility="collapsed")

        # ---- E ----
        section("E. What do you want the AI to do?", "Choose the kind of help you need.")
        field("Type of analysis", "Pick one from the list. Details are in the box underneath.", required=True)
        analysis_choice = st.selectbox("Type of analysis", list(ANALYSIS_TYPES.keys()), label_visibility="collapsed")
        with st.expander("ℹ️ What does each option give me?"):
            for name, info in ANALYSIS_TYPES.items():
                st.markdown(f"**{name}**: {info['help']}")
        field("Extra instructions", "Anything special to focus on, e.g. 'patient is pregnant' or 'focus on kidney safety'")
        st.text_input("Extra instructions", key="extra", placeholder="Optional", label_visibility="collapsed")

        st.write("")
        submitted = st.form_submit_button("✨ Analyze patient")

    # ---- Submit ----
    if submitted:
        s = st.session_state
        missing = []
        if not api_key:
            missing.append("**Groq API key** (Step 1, near the top of the page)")
        if s.get("age") is None:
            missing.append("**Age (years)** in section A")
        if not (s.get("complaint") or "").strip():
            missing.append("**Main complaint and symptoms** in section B")

        if missing:
            st.warning("Please complete these required items first:\n\n- " + "\n- ".join(missing), icon="📝")
        else:
            def fmt(v, unit=""):
                return f"{v}{unit}" if v not in (None, "") else ""

            patient = {
                "Age": fmt(s.get("age"), " years"),
                "Sex": "" if s.get("sex") == "Not stated" else s.get("sex"),
                "Weight": fmt(s.get("weight"), " kg"),
                "Main complaint and symptoms": (s.get("complaint") or "").strip(),
                "Heart rate": fmt(s.get("hr"), " beats/min"),
                "Blood pressure": (s.get("bp") or "").strip(),
                "Temperature": fmt(s.get("temp_c"), " °C"),
                "Oxygen saturation (SpO2)": fmt(s.get("spo2"), "%"),
                "Respiratory rate": fmt(s.get("rr"), " breaths/min"),
                "Past medical conditions": (s.get("history") or "").strip(),
                "Current medicines": (s.get("meds") or "").strip(),
                "Allergies": (s.get("allergies") or "").strip(),
                "Test results / imaging": (s.get("labs") or "").strip(),
                "Extra instructions": (s.get("extra") or "").strip(),
            }
            user_prompt = (
                f"Patient information:\n{build_patient_summary(patient)}\n\n"
                f"Task: {ANALYSIS_TYPES[analysis_choice]['task']}"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            st.markdown("### 📄 Step 3: Your results")
            st.caption(f"Analysis type: {analysis_choice}")
            try:
                client = Groq(api_key=api_key)
                with st.container(border=True):
                    with st.spinner("Reviewing the case. This usually takes a few seconds…"):
                        result = st.write_stream(stream_response(client, messages, model_id, temperature))

                report = (
                    "CareCompass report\n"
                    f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n"
                    f"Model: {model_id}\n"
                    f"Analysis: {analysis_choice}\n\n"
                    f"--- Patient information ---\n{build_patient_summary(patient)}\n\n"
                    f"--- AI output ---\n{result}\n\n"
                    "Decision support only. A clinician must verify this output."
                )
                st.download_button("⬇️ Download this report (.txt)", report,
                                   file_name="carecompass_report.txt", mime="text/plain")
            except Exception as e:
                st.error("Could not get a response from Groq. Check that your API key is correct "
                         "and still active, then press Analyze patient again.", icon="⚠️")
                st.caption(f"Technical details: {e}")

# ============================ TAB 2: Chat ===================================
with tab_chat:
    st.markdown("### 💬 Ask a clinical question")
    st.caption("Type a general question in the box at the very bottom of the page, for example "
               "'What are the warning signs of sepsis?'. This chat does not use the patient form.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    if st.button("🧹 Start a new conversation"):
        st.session_state.chat = []
        st.rerun()

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Type your question here and press Enter…")
    if question:
        if not api_key:
            st.warning("Please paste your Groq API key in Step 1 (top of the page) first.", icon="🔑")
        else:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.chat
            try:
                client = Groq(api_key=api_key)
                with st.chat_message("assistant"):
                    answer = st.write_stream(stream_response(client, messages, model_id, temperature))
                st.session_state.chat.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("Could not get a response. Please check your API key.", icon="⚠️")
                st.caption(f"Technical details: {e}")

# ============================ TAB 3: About ==================================
with tab_about:
    st.markdown(
        """
### 🗺️ Where do I enter what?

| What you want to enter | Where it is |
|---|---|
| Groq API key | **Step 1** box near the top of the page (not needed if already set in app secrets) |
| Age, sex, weight | **Patient analysis** tab, section A |
| Symptoms | **Patient analysis** tab, section B |
| Heart rate, blood pressure, temperature, oxygen, breathing rate | **Patient analysis** tab, section C |
| Conditions, medicines, allergies, test results | **Patient analysis** tab, section D |
| What kind of help you want | **Patient analysis** tab, section E |
| A general medical question | **Ask a question** tab, box at the bottom |
| AI model and answer style | **Left sidebar** (tap the small arrow at the top left on a phone) |

### ❓ Common problems
- **Nothing happens when I press Analyze:** read the yellow message. It lists which required items are missing.
- **"Could not get a response":** your API key may be wrong, expired or out of quota. Create a new one at console.groq.com.
- **Sidebar is hidden:** tap the **›** arrow at the top left.
- **Want to try the app first:** press **Fill with example patient** in the Patient analysis tab.

### ✅ Benefits
- Earlier spotting of risks and red flags
- A consistent second look that helps reduce cognitive bias
- Faster drafting of summaries and patient explanations

### ⚠️ Limitations
- AI can be wrong or sound confident when it is mistaken
- It only knows what you type in and cannot examine the patient
- Output may reflect bias in its training data
- It is not a certified medical device

### 🔒 Good practice
- Never enter names, ID numbers or other identifying details
- Treat every output as a suggestion, not an order
- Follow your hospital's AI governance and privacy policies
        """
    )
