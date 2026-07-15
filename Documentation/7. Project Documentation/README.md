# 7. Project Documentation

## Project Overview

Emotion Detection & Learning Support Engine is an AI-powered application designed to identify emotions from text and provide personalized learning guidance using deep learning and generative AI.

The system combines BiLSTM, BERT, and Google Gemini to enhance the learning experience.

---

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Emotion-Detection-Learning-Support.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Emotion-Detection-Learning-Support
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file and add:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
Emotion-Detection-Learning-Support/
│
├── Documentation/
├── models/
│   ├── bilstm/
│   └── bert_emotion_model_final/
│
├── screenshots/
├── app.py
├── requirements.txt
└── README.md
```

---

## Main Features

- Emotion detection using BiLSTM.
- Emotion detection using BERT.
- Personalized guidance using Gemini AI.
- Interactive dashboard.
- Prediction history.
- CSV export functionality.

---

## Technologies Used

- Python
- TensorFlow
- Hugging Face Transformers
- Streamlit
- Plotly
- Pandas
- NumPy
- Google Gemini API
- GitHub

---

## Future Improvements

- Improve BiLSTM performance.
- Add speech emotion recognition.
- Support multilingual emotion detection.
- Deploy the application to the cloud.

---

## Repository

Upload the project to GitHub and maintain version control using Git.
