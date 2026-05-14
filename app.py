import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict",
    page_icon="🏥",
    layout="centered"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0d1117; }

    .stApp {
        background-color: #0d1117;
        color: #e2e8f0;
    }

    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .title-block h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }

    .title-block p {
        color: #64748b;
        font-size: 1rem;
    }

    .result-card {
        background: #161923;
        border: 1px solid #1f2333;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .result-card.top {
        border-color: #10b981;
        background: #0d1f1a;
    }

    .disease-name {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    .confidence {
        font-size: 0.95rem;
        font-weight: 700;
        color: #10b981;
    }

    .rank-badge {
        background: #1f2333;
        color: #94a3b8;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 0.7rem;
    }

    .rank-badge.gold {
        background: #10b981;
        color: #ffffff;
    }

    .stTextArea textarea {
        background-color: #161923 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1f2333 !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.85;
    }

    .info-box {
        background: #161923;
        border: 1px solid #1f2333;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 1rem;
    }

    .divider {
        border: none;
        border-top: 1px solid #1f2333;
        margin: 1.5rem 0;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  NLP PIPELINE (copied from notebook)
# ─────────────────────────────────────────────
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your',
    'yours','yourself','yourselves','he','him','his','himself','she',
    'her','hers','herself','it','its','itself','they','them','their',
    'theirs','themselves','what','which','who','whom','this','that',
    'these','those','am','is','are','was','were','be','been','being',
    'have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of',
    'at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from',
    'up','down','in','out','on','off','over','under','again','further',
    'then','once','here','there','when','where','why','how','all',
    'both','each','few','more','most','other','some','such','no','nor',
    'not','only','own','same','so','than','too','very','s','t','can',
    'will','just','don','should','now','d','ll','m','o','re','ve','y'
}

LEMMA_MAP = {
    'itching': 'itch',    'itches': 'itch',     'itched': 'itch',
    'sweating': 'sweat',  'sweats': 'sweat',
    'vomiting': 'vomit',  'vomits': 'vomit',
    'bleeding': 'bleed',  'bleeds': 'bleed',
    'swelling': 'swell',  'swells': 'swell',    'swollen': 'swell',
    'burning':  'burn',   'burns':  'burn',
    'fatigue':  'fatigue','fatigued':'fatigue',
    'coughing': 'cough',  'coughs': 'cough',
    'breathing':'breathe','breathes':'breathe',
    'yellowing':'yellow', 'yellowed':'yellow',
    'rashes':   'rash',   'rashness':'rash',
    'pains':    'pain',   'painful': 'pain',
    'aches':    'ache',   'aching':  'ache',
    'dizziness':'dizzy',  'dizzying':'dizzy',
    'nausea':   'nausea', 'nauseous':'nausea',
    'infected': 'infect', 'infection':'infect',
}

def nlp_preprocess(symptom_list):
    tokens = []
    for sym in symptom_list:
        if sym and str(sym).strip():
            sym = str(sym).lower().strip()
            sym = re.sub(r'[_\-]', ' ', sym)
            sym = re.sub(r'[^a-z\s]', '', sym)
            words = sym.split()
            words = [w for w in words if w not in STOPWORDS and len(w) > 2]
            words = [LEMMA_MAP.get(w, w) for w in words]
            tokens.extend(words)
    return ' '.join(tokens)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    model  = pickle.load(open('model.pkl', 'rb'))
    tfidf  = pickle.load(open('tfidf.pkl', 'rb'))
    le     = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, tfidf, le

try:
    rf_model, tfidf, le = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🏥 MediPredict</h1>
    <p>Medical Symptom → Disease Prediction &nbsp;|&nbsp; NLP + Random Forest</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model files. Make sure `model.pkl`, `tfidf.pkl`, and `label_encoder.pkl` are in the same folder.\n\n**Error:** {load_error}")
    st.stop()

# Input
symptoms_input = st.text_area(
    "**Enter your symptoms:**",
    placeholder="e.g.  fever  chills  headache  nausea  vomiting",
    height=100
)

predict_btn = st.button("🔍 Predict Disease")

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
if predict_btn:
    if not symptoms_input.strip():
        st.warning("Please enter at least one symptom.")
    else:
        with st.spinner("Analysing symptoms..."):
            clean   = nlp_preprocess([symptoms_input])
            vec     = tfidf.transform([clean])
            proba   = rf_model.predict_proba(vec)[0]
            top3    = np.argsort(proba)[::-1][:3]

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("#### 🩺 Top Predictions")

        medals = ["🥇", "🥈", "🥉"]
        for rank, idx in enumerate(top3):
            disease    = le.inverse_transform([idx])[0]
            confidence = proba[idx] * 100
            card_class = "result-card top" if rank == 0 else "result-card"
            badge_class = "rank-badge gold" if rank == 0 else "rank-badge"

            st.markdown(f"""
            <div class="{card_class}">
                <div>
                    <span class="{badge_class}">{medals[rank]}</span>
                    <span class="disease-name">{disease}</span>
                </div>
                <span class="confidence">{confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only.
            Always consult a qualified medical professional for diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#334155; font-size:0.8rem;'>MediPredict · Nupur Zile · NLP + Random Forest</p>",
    unsafe_allow_html=True
)
