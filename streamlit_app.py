import streamlit as st
import pickle
import os
import re
import string

# --- File Paths ---
MODEL_DIR = 'model_files'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'nb_model.pkl')

# --- Check if model files exist ---
if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
    st.error("Model files not found. Please train the model by running 'retrain_model.py'.")
    st.stop()

# --- Load the saved model and vectorizer ---
try:
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- Preprocessing Function ---
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

# --- Streamlit App Interface ---
st.title("Fake News Detector")
st.markdown("Enter a news article or paragraph below to check if it's real or fake.")

user_input = st.text_area(
    "Paste your news text here:", 
    placeholder="For example: 'BREAKING: The moon has been confirmed to be made of cheese, according to a new government report.'",
    height=200
)

if st.button("Analyze News", type="primary"):
    if user_input:
        # Preprocess the input text
        processed_input = word_drop(user_input)

        # Vectorize the input using the trained vectorizer
        vectorized_input = vectorizer.transform([processed_input])

        # Get the prediction probabilities
        probabilities = model.predict_proba(vectorized_input)
        
        # Get the confidence for each class
        # The order of classes in model.classes_ is ['Fake', 'Real']
        # This code will use the probability with the highest value
        confidence_fake = probabilities[0][0] * 100
        confidence_real = probabilities[0][1] * 100
        
        # Determine the result based on the highest confidence
        if confidence_real > confidence_fake:
            result = "Real News"
        else:
            result = "Fake News"

        # Display the result with confidence scores
        st.markdown(f"### Prediction: :green[{result}]" if result == "Real News" else f"### Prediction: :red[{result}]")
        st.markdown(f"Confidence: Real News - {confidence_real:.2f}% | Fake News - {confidence_fake:.2f}%")
        st.markdown("---")

    else:
        st.warning("Please enter some text to analyze.")

st.caption("Powered by Streamlit and your trained machine learning model.")
