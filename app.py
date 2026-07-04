import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# -----------------------
# MODEL PATH
# -----------------------
MODEL_PATH = "models/bert_emotion_model_final"

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------
# SAFE LABEL DECODER
# -----------------------
def decode_label(idx):
    try:
        label = model.config.id2label

        # if proper dict exists
        if isinstance(label, dict):
            if idx in label:
                return str(label[idx])

        # fallback
        return f"emotion_{idx}"
    except:
        return f"emotion_{idx}"

# -----------------------
# FINAL EMOTION CLEANER
# -----------------------
def clean_emotion(label):
    label = str(label).lower()

    # remove common junk
    label = label.replace("label_", "")
    label = label.replace("emotion_", "")

    return label

# -----------------------
# EMOJI MAP
# -----------------------
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

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Emotion AI", page_icon="🤖")
st.title("🤖 Emotion AI Assistant (Fixed Version)")

if "history" not in st.session_state:
    st.session_state.history = []

text = st.text_area("Enter your text:")

# -----------------------
# PREDICT
# -----------------------
if st.button("Predict Emotion"):

    if text.strip():

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)[0]
        top3 = torch.topk(probs, 3)

        results = []

        for i in range(3):
            idx = top3.indices[i].item()
            score = top3.values[i].item()

            raw_label = decode_label(idx)
            emotion = clean_emotion(raw_label)

            results.append((emotion, score))

        top_emotion, top_score = results[0]
        emoji = get_emoji(top_emotion)

        # -----------------------
        # OUTPUT
        # -----------------------
        st.success(f"{emoji} Emotion: {top_emotion}")
        st.info(f"Confidence: {top_score * 100:.2f}%")

        # -----------------------
        # GRAPH
        # -----------------------
        st.subheader("Top 3 Emotion Probabilities 📊")
        df = pd.DataFrame(results, columns=["Emotion", "Confidence"])
        st.bar_chart(df.set_index("Emotion"))

        # -----------------------
        # HISTORY
        # -----------------------
        st.session_state.history.append((text, top_emotion, top_score))

    else:
        st.warning("Please enter some text")

# -----------------------
# HISTORY
# -----------------------
st.markdown("---")
st.subheader("History")

for t, e, c in reversed(st.session_state.history):
    st.write(f"{t} → {e} ({c*100:.2f}%)")

# -----------------------
# CLEAR HISTORY
# -----------------------
if st.button("Clear History"):
    st.session_state.history = []