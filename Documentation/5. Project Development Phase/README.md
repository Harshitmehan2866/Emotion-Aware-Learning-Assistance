# 5. Project Development Phase

## Overview

The Emotion Detection & Learning Support Engine was developed using deep learning, natural language processing, and generative AI technologies.

The development process involved dataset preparation, model training, frontend creation, and AI integration.

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed null and duplicate values.
- Cleaned text data.
- Converted text to lowercase.
- Removed special characters and punctuation.
- Tokenized text.
- Encoded emotion labels.
- Applied sequence padding.

---

## Dataset Statistics

Datasets used:

- GoEmotions
- ISEAR
- Empathetic Dialogues

Combined dataset size:

- Total samples: 132,125
- Emotion classes: 59

---

## BiLSTM Model Development

Model architecture:

- Embedding Layer
- Bidirectional LSTM Layer
- Dense Layers
- Dropout Layers
- Softmax Output Layer

Training configuration:

- Sequence length: 100
- Vocabulary size: 40,000
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

---

## BERT Model Development

Features:

- Transformer-based architecture.
- Context-aware emotion classification.
- Fine-tuned using Hugging Face Transformers.

Training components:

- Tokenizer
- Attention masks
- Classification head

---

## Gemini AI Integration

Google Gemini API was integrated to provide:

- Personalized learning guidance.
- Emotion-based recommendations.
- Motivational feedback.

---

## Streamlit Application

Frontend features:

- Text input area.
- Study field selection.
- Emotion prediction.
- AI-generated guidance.
- Analytics dashboard.
- Prediction history.
- CSV export.

---

## Libraries Used

```python
tensorflow
transformers
streamlit
torch
pandas
numpy
plotly
scikit-learn
google-generativeai
```

---

## Source Code Management

Version control was maintained using:

- Git
- GitHub

Repository structure was organized to support scalability and maintainability.