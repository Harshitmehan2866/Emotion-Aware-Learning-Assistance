import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Emotion-Aware Learning Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Emotion-Aware Learning Assistant")

st.markdown(
    """
This application uses a fine-tuned BERT model to detect emotions
from text and provide an interactive learning experience.
"""
)

# =====================================================
# MODEL PATH
# =====================================================

MODEL_PATH = "models/bert_emotion_model_final"

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🧠 Dashboard")

st.sidebar.success("✅ BERT Model Loaded")

st.sidebar.metric(
    "Predictions Made",
    len(st.session_state.history)
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
Model: BERT

Framework: Transformers

Frontend: Streamlit
"""
)

# =====================================================
# LABEL DECODER
# =====================================================

emotion_labels = [
    'admiration',
    'afraid',
    'amusement',
    'anger',
    'angry',
    'annoyance',
    'annoyed',
    'anticipating',
    'anxious',
    'apprehensive',
    'approval',
    'ashamed',
    'caring',
    'confident',
    'confusion',
    'content',
    'curiosity',
    'desire',
    'devastated',
    'disappointed',
    'disappointment',
    'disapproval',
    'disgust',
    'disgusted',
    'embarrassed',
    'embarrassment',
    'excited',
    'excitement',
    'faithful',
    'fear',
    'furious',
    'grateful',
    'gratitude',
    'grief',
    'guilty',
    'hopeful',
    'impressed',
    'jealous',
    'joy',
    'joyful',
    'lonely',
    'love',
    'nervousness',
    'neutral',
    'nostalgic',
    'optimism',
    'prepared',
    'pride',
    'proud',
    'realization',
    'relief',
    'remorse',
    'sad',
    'sadness',
    'sentimental',
    'surprise',
    'surprised',
    'terrified',
    'trusting'
]

def decode_label(idx):

    if 0 <= idx < len(emotion_labels):
        return emotion_labels[idx]

    return "unknown"


# =====================================================
# CLEAN LABEL
# =====================================================

def clean_emotion(label):

    label = str(label).lower()

    label = label.replace("label_", "")

    label = label.replace("emotion_", "")

    return label


# =====================================================
# EMOJI MAP
# =====================================================

def get_emoji(emotion):

    emotion = emotion.lower()

    if "joy" in emotion or "happy" in emotion:
        return "😄"

    if "anger" in emotion:
        return "😡"

    if "fear" in emotion:
        return "😨"

    if "love" in emotion:
        return "❤️"

    if "sad" in emotion:
        return "😢"

    if "surprise" in emotion:
        return "😲"

    return "🙂"


# =====================================================
# USER INPUT
# =====================================================

st.subheader("✍️ Enter Text")

text = st.text_area(
    "Write your message below:",
    height=180,
    placeholder="Example: I am feeling very excited about my exams."
)

# =====================================================
# PREDICTION
# =====================================================

if st.button(
    "🚀 Predict Emotion",
    use_container_width=True
):

    if text.strip():

        with st.spinner("Analyzing emotion..."):

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            with torch.no_grad():

                outputs = model(**inputs)

            probs = F.softmax(
                outputs.logits,
                dim=1
            )[0]

            top3 = torch.topk(
                probs,
                3
            )

            results = []

            for i in range(3):

                idx = top3.indices[i].item()

                score = top3.values[i].item()

                raw_label = decode_label(idx)

                emotion = clean_emotion(
                    raw_label
                )

                results.append(
                    (
                        emotion,
                        score
                    )
                )

            top_emotion, top_score = results[0]

            emoji = get_emoji(
                top_emotion
            )

        st.success(
            "✅ Emotion detected successfully"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"{emoji} Emotion"
            )

            st.metric(
                "Detected Emotion",
                top_emotion.title()
            )

        with col2:

            st.info(
                "Prediction Confidence"
            )

            st.metric(
                "Confidence",
                f"{top_score * 100:.2f}%"
            )

        # ==========================================
        # GRAPH
        # ==========================================

        st.markdown("---")

        st.subheader(
            "📊 Top 3 Emotion Probabilities"
        )

        df = pd.DataFrame(
            results,
            columns=[
                "Emotion",
                "Confidence"
            ]
        )

        st.bar_chart(
            df.set_index(
                "Emotion"
            )
        )

        # ==========================================
        # SAVE HISTORY
        # ==========================================

        st.session_state.history.append(
    (
        text,
        top_emotion.title(),
        round(top_score * 100, 2)
    )
)

    else:

        st.warning(
            "Please enter some text."
        )

# =====================================================
# HISTORY
# =====================================================

st.markdown("---")

st.subheader(
    "📜 Prediction History"
)

if len(
    st.session_state.history
) > 0:

    history_df = pd.DataFrame(

        st.session_state.history,

        columns=[
            "Text",
            "Emotion",
            "Confidence (%)"
        ]
    )

    st.dataframe(

        history_df,

        use_container_width=True

    )

    csv = history_df.to_csv(

        index=False

    ).encode("utf-8")

    st.download_button(

        "⬇ Download History",

        csv,

        "prediction_history.csv",

        "text/csv"

    )

else:

    st.info(
        "No predictions yet."
    )

# =====================================================
# CLEAR HISTORY
# =====================================================

if st.button(
    "🗑 Clear History"
):

    st.session_state.history = []

    st.rerun()

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center;color:gray;'>

<h4>🧠 Emotion-Aware Learning Assistant</h4>

<p>Powered by BERT • Transformers • Streamlit</p>

<p><b>Developed by Harshit Mehan</b></p>

</div>
""",
    unsafe_allow_html=True
)