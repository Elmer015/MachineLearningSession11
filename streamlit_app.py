from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import streamlit as st


APP_TITLE = "Spam Detector"
APP_SUBTITLE = "Enter a message and check whether it is spam."
PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_PATH = PROJECT_ROOT / "spam_model.joblib"
METADATA_PATH = PROJECT_ROOT / "spam_model_metadata.json"

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH.resolve()}")

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def normalize_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = URL_PATTERN.sub(" URL ", cleaned)
    cleaned = EMAIL_PATTERN.sub(" EMAIL ", cleaned)
    cleaned = re.sub(r"[^\w\s@.+-]", " ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned


def predict_text(model, text: str) -> dict:
    normalized_text = normalize_text(text)
    prediction = int(model.predict([normalized_text])[0])
    probability = float(model.predict_proba([normalized_text])[0][1])
    return {
        "normalized_text": normalized_text,
        "prediction": "spam" if prediction == 1 else "ham",
        "spam_probability": probability,
    }


st.set_page_config(page_title=APP_TITLE, page_icon="📨", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(29, 78, 216, 0.14), transparent 30%),
            radial-gradient(circle at right top, rgba(15, 118, 110, 0.12), transparent 25%),
            linear-gradient(180deg, #f8fafc 0%, #eef4fb 100%);
    }
    .hero {
        padding: 1.5rem 1.75rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        background: rgba(255,255,255,0.88);
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.5rem 0 0 0;
        color: #475569;
        font-size: 1rem;
        line-height: 1.5;
    }
    .soft-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <div class="soft-label">Pretrained Spam Checker</div>
      <h1>{APP_TITLE}</h1>
      <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Failed to load model or metadata: {exc}")
    st.stop()

st.caption(f"Dataset type: {metadata.get('dataset_kind', 'unknown')}")

message = st.text_area(
    "Message",
    height=180,
    placeholder="Example: You won a big prize, click this link now",
)

if st.button("Check Spam", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        result = predict_text(model, message)
        if result["prediction"] == "spam":
            st.error(f"Result: SPAM | Spam probability: {result['spam_probability']:.3f}")
        else:
            st.success(f"Result: HAM / not spam | Spam probability: {result['spam_probability']:.3f}")
        st.write("Normalized text:")
        st.code(result["normalized_text"])

st.caption("The model is loaded from notebook artifacts. There is no training in this UI.")
