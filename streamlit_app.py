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
        probabilities = model.predict_proba(vectorized_input)[0]
        
        # Get the class labels from the model
        class_labels = model.classes_
        st.write(f"Model class order: {class_labels}")  # Debugging output

        # Map probabilities to class labels dynamically
        class_prob_dict = dict(zip(class_labels, probabilities))
        confidence_fake = class_prob_dict.get('Fake', 0) * 100
        confidence_real = class_prob_dict.get('Real', 0) * 100

        # Determine the result based on the highest probability
        predicted_class = max(class_prob_dict, key=class_prob_dict.get)
        confidence = class_prob_dict[predicted_class] * 100

        # Display the result with confidence scores
        st.markdown(f"### Prediction: :green[{predicted_class}]" if predicted_class == "Real" else f"### Prediction: :red[{predicted_class}]")
        st.markdown(f"Confidence: Real News - {confidence_real:.2f}% | Fake News - {confidence_fake:.2f}%")
        
        # Optional: Warn if confidence is low
        if confidence < 60:
            st.warning("The prediction confidence is low. The result may be uncertain.")
        
        st.markdown("---")
        
        # Debugging: Show processed input
        st.markdown("**Processed Input (for debugging):**")
        st.write(processed_input)

    else:
        st.warning("Please enter some text to analyze.")

st.caption("Powered by Streamlit and your trained machine learning model.")
