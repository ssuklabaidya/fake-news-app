# retrain_model.py
# This script loads your original dataset, retrains the model, and saves the new,
# correctly trained .pkl files.

import pandas as pd
import re
import string
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Starting model retraining script...")

# --- USER INPUT ---
# The folder where your CSV files are located
DATA_DIR = 'true1'

# The names of your CSV files
FAKE_NEWS_CSV = 'Fake.csv'
TRUE_NEWS_CSV = 'True.csv'

# Column names in your CSV files. Please change these if they are different.
TEXT_COLUMN = 'text' 
LABEL_COLUMN = 'class'

# --- 1. Load your original datasets ---
try:
    fake_df = pd.read_csv(os.path.join(DATA_DIR, FAKE_NEWS_CSV))
    true_df = pd.read_csv(os.path.join(DATA_DIR, TRUE_NEWS_CSV))
    
    print(f"Dataset '{FAKE_NEWS_CSV}' loaded with {len(fake_df)} rows.")
    print(f"Dataset '{TRUE_NEWS_CSV}' loaded with {len(true_df)} rows.")

    # Assign class labels: 0 for fake, 1 for true
    fake_df[LABEL_COLUMN] = 0
    true_df[LABEL_COLUMN] = 1

    # Combine the datasets
    data = pd.concat([fake_df, true_df], ignore_index=True)
    
    print(f"Combined dataset has {len(data)} rows.")

except FileNotFoundError:
    print(f"Error: One or both of the CSV files were not found in the '{DATA_DIR}' directory.")
    print("Please make sure your CSV files are in the specified folder.")
    exit()
except KeyError as e:
    print(f"Error: One of the specified columns was not found in the CSV files: {e}")
    print("Please check the TEXT_COLUMN and LABEL_COLUMN variables in the script.")
    exit()


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
data[TEXT_COLUMN] = data[TEXT_COLUMN].apply(word_drop)

print("Text preprocessing completed.")

# --- 3. Train the TF-IDF Vectorizer and Model ---
# This is the same process as in your notebook, but with your real data.
x = data[TEXT_COLUMN]
y = data[LABEL_COLUMN]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_vectorized = tfidf_vectorizer.fit_transform(x_train)
x_test_vectorized = tfidf_vectorizer.transform(x_test)

nb_model = MultinomialNB(alpha=0.2)
nb_model.fit(x_train_vectorized, y_train)
y_pred = nb_model.predict(x_test_vectorized)

print(f"Model trained with accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# --- 4. Save the Model and Vectorizer to the correct path ---
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

print("Model retraining script finished.")

