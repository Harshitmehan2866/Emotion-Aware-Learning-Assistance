# 3. Project Design Phase

## System Architecture

The Emotion Detection & Learning Support Engine follows a modular architecture consisting of data preprocessing, deep learning models, AI guidance generation, and a user-friendly interface.

---

## System Workflow

```text
User Input
     │
     ▼
Text Preprocessing
     │
     ▼
┌─────────────┬─────────────┐
│             │             │
▼             ▼             ▼
BiLSTM      BERT      Gemini AI
│             │             │
└─────────────┴─────────────┘
     │
     ▼
Emotion Prediction
     │
     ▼
Personalized Guidance
     │
     ▼
Analytics Dashboard
```

---

## Components

### 1. Data Preprocessing

- Text cleaning
- Tokenization
- Sequence padding
- Label encoding

### 2. BiLSTM Model

- Embedding Layer
- Bidirectional LSTM Layer
- Dense Layers
- Softmax Classification

### 3. BERT Model

- Transformer-based emotion classification
- Fine-tuned on emotion datasets
- Context-aware predictions

### 4. Gemini AI Module

- Generates personalized learning suggestions.
- Provides motivational guidance.

### 5. Streamlit Frontend

- User input form
- Visualization dashboard
- Prediction history
- CSV export

---

## Technologies Used

- Python
- TensorFlow
- Hugging Face Transformers
- Streamlit
- Plotly
- Pandas
- Scikit-learn
- Google Gemini API

---

## Folder Structure

```text
Emotion-Detection-Learning-Support/
│
├── models/
│   ├── bilstm/
│   └── bert_emotion_model_final/
│
├── Documentation/
│
├── screenshots/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Design Goals

- High accuracy emotion detection.
- Easy-to-use interface.
- Real-time predictions.
- Personalized learning support.
- Scalable architecture.