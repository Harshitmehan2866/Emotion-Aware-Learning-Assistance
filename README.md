# 🤖 Emotion AI Assistant using BERT (Fine-Tuned)

## 📌 Project Overview
This project is an AI-powered **Emotion Detection System** built using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model.  
The model is trained on an emotion classification dataset and deployed using **Streamlit** for real-time predictions.

The system predicts emotions from text and visualizes confidence scores for better interpretability.

---

## 🚀 Features

- 🔍 Real-time emotion detection from user input text  
- 🤖 Fine-tuned BERT transformer model  
- 📊 Top-3 emotion probability visualization  
- 😄 Emoji-based emotion output  
- 📜 Prediction history tracking  
- 🧹 Clear history option  
- ⚡ Fast inference using cached model loading  
- 📈 Visual insights using bar charts  

---

## 🧠 Emotion Classes

- Anger 😡  
- Fear 😨  
- Joy 😄  
- Love ❤️  
- Sadness 😢  

*(Based on training dataset labels)*

---

## 🏗️ Tech Stack

- Python 🐍  
- PyTorch 🔥  
- HuggingFace Transformers 🤗  
- Streamlit 🌐  
- Pandas 📊  
- Matplotlib 📈  

---

## 📂 Project Structure

```plaintext
Emotion-AI-Project/
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
│
├── models/
│   └── bert_emotion_model_final/  # Fine-tuned BERT model
│
├── training/
│   └── bert_training.ipynb        # Training notebook (all epochs included here)
│
└── images/
    └── demo.png                   # Optional UI screenshot
🧠 Model Training Details
📌 Model Used
BERT Base Uncased (Fine-tuned for classification)
📌 Dataset
Emotion classification dataset (e.g., GoEmotions / custom labeled dataset)
📊 Training Summary (IMPORTANT)

The model was trained for multiple epochs to improve convergence and accuracy:

Epoch	Loss (approx)	Observation
1	1.20	High loss, initial learning
2	0.85	Model starts learning patterns
3	0.62	Improved stability
4	0.45	Better classification boundaries
5	0.30	Final convergence achieved

✔ Final model selected from last epoch due to best validation performance.

📈 Performance
Improved accuracy after each epoch
Stable convergence after epoch 4
Final model shows strong generalization on unseen text
⚙️ How to Run Locally
1️⃣ Clone repository
git clone https://github.com/your-username/emotion-ai-project.git
cd emotion-ai-project
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run Streamlit app
streamlit run app.py
📊 Example Output

Input:

I am extremely happy today, everything worked out perfectly!

Output:

😄 Emotion: Joy  
Confidence: 93.21%
🚀 Key Improvements
Top-3 emotion probability display
Softmax-based confidence scoring
Emoji-enhanced UI
History tracking system
Clean transformer-based pipeline
🔮 Future Enhancements
🎤 Voice-based emotion detection
💬 Chatbot-style interface
🌐 Cloud deployment (Streamlit / HuggingFace Spaces)
📊 Advanced analytics dashboard
🧠 Multi-language emotion detection
👨‍💻 Author

Harshit Mehan
B.Tech CSE Student
AI / ML Enthusiast | Deep Learning Explorer