import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime

# -------------------------------------------------
# 1️⃣ PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="DS03 – Smart Manufacturing GenAI Assistant",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>🏭 Smart Manufacturing: GenAI Operations Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>AI-powered assistant for IoT-based machine monitoring, "
    "predictive maintenance, and operational decision support.</p>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# 2️⃣ GEMINI API CONFIG (REST)
# -------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets.")
    st.stop()

GEMINI_KEY = st.secrets["GEMINI_API_KEY"]

# -------------------------------------------------
# 3️⃣ LOAD DATA & MODELS
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("StreamlitApp/smart_manufacturing_data_latest.csv.gz")

@st.cache_resource
def load_models():
    model = joblib.load("StreamlitApp/final_predictive_maintenance_model.pkl")
    scaler = joblib.load("StreamlitApp/scaler.pkl")
    return model, scaler

df = load_data()
ml_model, scaler = load_models()

# -------------------------------------------------
# 4️⃣ BUILD ML CONTEXT (ANTI-HALLUCINATION)
# -------------------------------------------------
def build_ml_context(machine_id: str, df: pd.DataFrame) -> str:
    row = df[df["machine_id"] == machine_id].iloc[-1]

    failure_risk = "High" if row["maintenance_required"] == 1 else "Low"
    anomaly = "Yes" if row.get("anomaly_flag_pred", 0) == 1 else "No"

    context = f"""
    Machine ID: {machine_id}
    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Predictive Maintenance Results:
    - Failure Risk Level: {failure_risk}
    - Anomaly Detected: {anomaly}

    Rules:
    - High risk means maintenance may be required soon.
    - Anomaly means abnormal sensor behavior was detected.
    """

    return context.strip()

# -------------------------------------------------
# 5️⃣ GENAI SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = """
You are a Smart Manufacturing Operations Assistant.

Rules:
- Use ONLY the machine data provided.
- Do NOT invent sensor values.
- Do NOT mention AI, ML, or algorithms.
- Explain insights in simple operational language.
- Focus on business impact and maintenance action.
"""

# -------------------------------------------------
# 6️⃣ GEMINI REST API CALL
# -------------------------------------------------
def genai_answer(machine_id: str, user_question: str, df: pd.DataFrame) -> str:
    context = build_ml_context(machine_id, df)

    user_prompt = f"""
    MACHINE DATA:
    {context}

    USER QUESTION:
    "{user_question}"

    Answer using:
    1. What is happening
    2. Why it matters
    3. Recommended action
    """

    api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\n" + user_prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 700
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        return result["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return f"❌ GenAI response failed: {e}"

# -------------------------------------------------
# 7️⃣ STREAMLIT UI
# -------------------------------------------------
st.subheader("🔍 Machine Health & Predictive Intelligence")

col1, col2 = st.columns([1, 2])

with col1:
    machine_ids = sorted(df["machine_id"].unique())
    selected_machine = st.selectbox("Select Machine ID", machine_ids)

    st.markdown("#### Example Questions")
    st.markdown("""
    - Which machines are at risk today?
    - Why is this machine flagged?
    - What action is recommended?
    - Is immediate maintenance required?
    """)

with col2:
    user_question = st.text_input(
        "Ask your question",
        placeholder="e.g. Why is this machine flagged?"
    )

    if st.button("Ask GenAI Assistant"):
        if not user_question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing machine health..."):
                answer = genai_answer(selected_machine, user_question, df)

            st.markdown("### 🧠 GenAI Insight")
            st.markdown(answer)

# -------------------------------------------------
# 8️⃣ RAW ML OUTPUT (TRANSPARENCY)
# -------------------------------------------------  
with st.expander("🔎 View Raw ML Status"):
    latest = df[df["machine_id"] == selected_machine].iloc[-1]

    maintenance_text = (
        "Maintenance Required"
        if int(latest["maintenance_required"]) == 1
        else "No Maintenance Required"
    )

    anomaly_text = (
        "Anomaly Detected"
        if int(latest.get("anomaly_flag_pred", 0)) == 1
        else "No Anomaly Detected"
    )

    st.markdown(f"""
    **Machine ID:** {selected_machine}  

    **Maintenance Status:** {maintenance_text}  

    **Anomaly Status:** {anomaly_text}
    """)


# -------------------------------------------------
# 9️⃣ SIDEBAR PROJECT INFO
# -------------------------------------------------
with st.sidebar:
    st.header("📌 Project Info")
    st.markdown("""
    **DS03: Smart Manufacturing**

    - IoT Sensor Monitoring  
    - Predictive Maintenance (ML)  
    - GenAI Explanation Layer  
    - Streamlit Deployment  

    **Goal:**  
    Enable plant managers to take fast, informed maintenance decisions.
    """)

