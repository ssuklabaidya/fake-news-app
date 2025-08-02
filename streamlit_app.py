# streamlit_app.py

import streamlit as st
import pickle
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# --- Global variables and model loading ---
# Update these paths to point to your new folder structure
MODEL_PATH = 'model_files/nb_model.pkl'
VECTORIZER_PATH = 'model_files/vectorizer.pkl'

# --- Preprocessing function (same as before) ---
def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# --- Load the model and vectorizer at the start ---
# Use st.cache_resource to load these heavy objects only once
@st.cache_resource
def load_resources():
    try:
        # Check if files exist before trying to load
        if not os.path.exists(MODEL_PATH):
            st.error(f"Error: Model file not found at '{MODEL_PATH}'.")
            st.stop()
        if not os.path.exists(VECTORIZER_PATH):
            st.error(f"Error: Vectorizer file not found at '{VECTORIZER_PATH}'.")
            st.stop()

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or vectorizer file not found.")
        st.stop()

fake_news_model, tfidf_vectorizer = load_resources()

# --- Streamlit UI and Prediction Logic ---
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    .header-text {
        font-size: 3rem;
        font-weight: 800;
        color: #1a237e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subheader-text {
        font-size: 1.25rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box-fake {
        background-color: #fce7e7;
        color: #c53030;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #e53e3e;
    }
    .result-box-real {
        background-color: #e6fffa;
        color: #2c7a7b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #38b2ac;
    }
    .result-text {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .footer-text {
        text-align: center;
        font-size: 0.8rem;
        color: #a0aec0;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header-text">🤖 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Enter a news article or paragraph below to check if it\'s real or fake.</p>', unsafe_allow_html=True)

user_input = st.text_area(
    "Paste your news text here:",
    height=200,
    placeholder="For example: 'BREAKING: The moon has been confirmed to be made of cheese, according to a new government report.'",
    help="The model will analyze the text and give you a prediction."
)

if st.button("Analyze News", use_container_width=True, type="primary"):
    if user_input:
        processed_text = [word_drop(user_input)]
        
        vectorized_text = tfidf_vectorizer.transform(processed_text)
        
        prediction_label = fake_news_model.predict(vectorized_text)[0]
        prediction_proba = fake_news_model.predict_proba(vectorized_text)[0]
        
        prediction_dict = {0: "Fake News", 1: "Real News"}
        predicted_class = prediction_dict[prediction_label]
        
        if predicted_class == "Fake News":
            st.markdown(f"""
                <div class="result-box-fake">
                    <p class="result-text">Prediction: {predicted_class}</p>
                    <p class="confidence-text">Confidence: Fake News - {prediction_proba[0]*100:.2f}% | Real News - {prediction_proba[1]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box-real">
                    <p class="result-text">Prediction: {predicted_class}</p>
                    <p class="confidence-text">Confidence: Real News - {prediction_proba[1]*100:.2f}% | Fake News - {prediction_proba[0]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown('<p class="footer-text">Powered by Streamlit and your trained machine learning model.</p>', unsafe_allow_html=True)

