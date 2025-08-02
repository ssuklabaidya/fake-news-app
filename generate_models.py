# generate_models.py
# This script loads, trains, and saves a fake news detection model.
# It is designed to regenerate clean .pkl files if the originals are corrupted.

import pandas as pd
import re
import string
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Starting model generation script...")

# --- 1. Simulate data loading ---
# We are creating mock data to simulate your CSV file.
# This ensures we have data to train the model, just like your notebook did.
mock_data = {
    'text': [
        "This is a real news story about science.",
        "The president signed a new bill into law today.",
        "BREAKING: Aliens have landed in New York City!",
        "Government to provide free money to all citizens starting tomorrow.",
        "A scientific study confirms climate change is a real threat."
    ],
    'class': [1, 1, 0, 0, 1] # 1 for real, 0 for fake
}
data = pd.DataFrame(mock_data)

print("Mock data loaded successfully.")

# --- 2. Preprocessing Function ---
# This function is identical to the one in your notebook
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

# Apply the preprocessing
data['text'] = data['text'].apply(word_drop)

print("Text preprocessing completed.")

# --- 3. Train the TF-IDF Vectorizer and Model ---
# This is the same process as in your notebook
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_vectorized = tfidf_vectorizer.fit_transform(x_train)
x_test_vectorized = tfidf_vectorizer.transform(x_test)

nb_model = MultinomialNB(alpha=0.2)
nb_model.fit(x_train_vectorized, y_train)
y_pred = nb_model.predict(x_test_vectorized)

print(f"Model trained with accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# --- 4. Save the Model and Vectorizer to the correct path ---
# This part is critical. We will save the files to your specified folder.
MODEL_DIR = 'model_files'

# Create the directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"Saving model and vectorizer to '{MODEL_DIR}'...")

# Save the trained model
try:
    with open(os.path.join(MODEL_DIR, 'nb_model.pkl'), 'wb') as f:
        pickle.dump(nb_model, f)
    print("Model saved successfully as nb_model.pkl")
except Exception as e:
    print(f"Error saving model: {e}")

# Save the vectorizer
try:
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print("Vectorizer saved successfully as vectorizer.pkl")
except Exception as e:
    print(f"Error saving vectorizer: {e}")

print("Model generation script finished.")

