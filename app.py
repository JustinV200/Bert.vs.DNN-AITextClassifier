import torch
import pickle
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import torch.nn.functional as F
import os
from flask import Flask, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models and tokenizers once at startup
dnn_model = load_model('DNN_results/ai_vs_human_classifier.keras')
with open('DNN_results/tokenizer.pickle', 'rb') as f:
    dnn_tokenizer = pickle.load(f)

bert_tokenizer = BertTokenizer.from_pretrained('Bert_results')
bert_model = BertForSequenceClassification.from_pretrained('Bert_results')

# #same cleaning function as in TextClassifierDNN.p and TextClassifierBERT.py, nothing new
def clean_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    text = re.sub(r'<[^>]+>', '', text)
    return text

# #predict function that uses both models
# #returns a dictionary with the predictions from both models
def compare_models(text):
    cleaned = clean_text(text)

    # DNN prediction
    dnn_seq = dnn_tokenizer.texts_to_sequences([cleaned])
    dnn_pad = pad_sequences(dnn_seq, padding='post', maxlen=300)
    dnn_pred = dnn_model.predict(dnn_pad)[0][0]
    dnn_label = "Human" if dnn_pred > 0.5 else "AI"

    # BERT prediction with confidence
    tokens = bert_tokenizer(cleaned, padding="max_length", truncation=True, max_length=300, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
        logits = output.logits
        probs = F.softmax(logits, dim=1)
        confidence = torch.max(probs).item()
        bert_pred = torch.argmax(logits, dim=1).item()
        bert_label = "Human" if bert_pred == 1 else "AI"

    return {
        "dnn": {"label": dnn_label, "confidence": float(dnn_pred)},
        "bert": {"label": bert_label, "confidence": confidence, "logits": logits.tolist()[0]}
    }

# Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        result = compare_models(text)
    return render_template('index.html', result=result)

