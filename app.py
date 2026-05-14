import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediSense — Drug Review Sentiment",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #e2e8f0; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #161923;
        border: 1px solid #1f2333;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .positive-card {
        background: #0f2e1f;
        border: 1px solid #1d9e75;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .negative-card {
        background: #2e0f0f;
        border: 1px solid #e24b4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stTextArea textarea {
        background-color: #161923 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1f2333 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: #185FA5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: #378ADD; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("drug_sentiment_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Text cleaning (same as notebook) ───────────────────────────────
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = str(text)
    text = re.sub(r'&#\d+;', "'", text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 MediSense")
    st.markdown("**Drug Review Sentiment Classifier**")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app predicts whether a patient drug review is **Positive** or **Negative** using an NLP pipeline.
    """)
    st.markdown("### Model Info")
    st.markdown("""
    - **Algorithm:** Logistic Regression
    - **Features:** TF-IDF (unigrams + bigrams)
    - **Vocab size:** 15,000 terms
    - **Accuracy:** ~92–94%
    """)
    st.markdown("### Sentiment Rules")
    st.markdown("""
    - ⭐ Rating ≥ 7 → **Positive**
    - ⭐ Rating ≤ 4 → **Negative**
    """)
    st.markdown("---")
    st.markdown("**Author:** Nupur Zile")

# ── Main Page ──────────────────────────────────────────────────────
st.markdown("# 💊 MediSense — Drug Review Sentiment Analysis")
st.markdown("Enter a patient drug review below and the model will predict whether it is **Positive** or **Negative**.")
st.markdown("---")

# ── Input + Predict ────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✍️ Enter Drug Review")
    review_input = st.text_area(
        label="",
        placeholder="e.g. This medication completely changed my life. No side effects, works great! I feel much better after 2 weeks...",
        height=160,
        key="review_input"
    )

    predict_btn = st.button("🔍 Predict Sentiment", use_container_width=True)

with col2:
    st.markdown("### 💡 Try Sample Reviews")
    samples = {
        "😊 Positive Example": "This medication completely changed my life. No side effects, works great! I feel much better after 2 weeks.",
        "😔 Negative Example": "Terrible drug. Made me sick and dizzy every single day. Constant nausea and headaches. Stopped after one week.",
        "😊 Positive Example 2": "After 3 weeks my condition improved significantly. Doctor was right. Highly recommend this medicine.",
        "😔 Negative Example 2": "Horrible experience. Severe side effects and no improvement in my condition. Would not recommend.",
    }
    for label, text in samples.items():
        if st.button(label, key=label):
            st.session_state["review_input"] = text
            st.rerun()

st.markdown("---")

# ── Prediction Result ──────────────────────────────────────────────
if predict_btn and review_input.strip():
    cleaned = clean_text(review_input)

    prediction  = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned])[0]
    classes     = list(model.classes_)
    pos_prob    = probability[classes.index('positive')]
    neg_prob    = probability[classes.index('negative')]

    st.markdown("### 📊 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 'positive':
            st.markdown(f"""
            <div class="positive-card">
                <h2 style="color:#1d9e75;margin:0">✅ POSITIVE</h2>
                <p style="color:#9fe1cb;margin:8px 0 0">Sentiment Prediction</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="negative-card">
                <h2 style="color:#e24b4a;margin:0">❌ NEGATIVE</h2>
                <p style="color:#f09595;margin:8px 0 0">Sentiment Prediction</p>
            </div>""", unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color:#1d9e75;margin:0">{pos_prob:.1%}</h2>
            <p style="color:#9fe1cb;margin:8px 0 0">Positive Confidence</p>
        </div>""", unsafe_allow_html=True)

    with res_col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color:#e24b4a;margin:0">{neg_prob:.1%}</h2>
            <p style="color:#f09595;margin:8px 0 0">Negative Confidence</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Confidence Score")
    st.progress(float(pos_prob), text=f"Positive: {pos_prob:.1%}")

    with st.expander("🔍 See Cleaned Text (after preprocessing)"):
        st.code(cleaned, language=None)

elif predict_btn and not review_input.strip():
    st.warning("⚠️ Please enter a drug review before clicking Predict.")

# ── Batch Prediction ───────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Batch Prediction")
st.markdown("Paste multiple reviews (one per line) to predict all at once.")

batch_input = st.text_area(
    label="",
    placeholder="Review 1: This drug helped me a lot...\nReview 2: Worst medication ever...\nReview 3: ...",
    height=120,
    key="batch_input"
)

if st.button("🔍 Predict All Reviews", key="batch_btn"):
    if batch_input.strip():
        lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]
        cleaned_lines = [clean_text(l) for l in lines]
        preds  = model.predict(cleaned_lines)
        probas = model.predict_proba(cleaned_lines)
        classes = list(model.classes_)

        st.markdown("#### Results:")
        for i, (line, pred, prob) in enumerate(zip(lines, preds, probas)):
            pos_p = prob[classes.index('positive')]
            icon  = "✅" if pred == 'positive' else "❌"
            color = "#1d9e75" if pred == 'positive' else "#e24b4a"
            st.markdown(f"""
            <div style="background:#161923;border:1px solid #1f2333;border-radius:8px;padding:12px 16px;margin:8px 0">
                <span style="font-size:12px;color:#64748b">Review {i+1}</span><br>
                <span style="color:#e2e8f0;font-size:13px">{line[:120]}{'...' if len(line)>120 else ''}</span><br>
                <span style="color:{color};font-weight:600">{icon} {pred.upper()}</span>
                <span style="color:#64748b;font-size:12px;margin-left:12px">Positive: {pos_p:.1%}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("Please enter at least one review.")

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#475569;font-size:12px;padding:10px 0">
    MediSense — Drug Review Sentiment Analysis | Built by Nupur Zile | NLP + Logistic Regression + TF-IDF
</div>
""", unsafe_allow_html=True)
