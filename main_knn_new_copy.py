# main_knn.py

import joblib
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import lmstudio as lms


import os
import sys


def resource_path(relative_path):
    """ Get absolute path to resource, works for both dev and PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores files there
        base_path = sys._MEIPASS
    except AttributeError:
        # If not bundled, use the current directory
        base_path = os.path.abspath(".")

    # Ensure relative_path is a string
    if not isinstance(relative_path, str):
        raise TypeError(f"Expected a string for relative_path, got {type(relative_path)}: {relative_path}")

    # Join the base path and relative path
    absolute_path = os.path.join(base_path, relative_path)

    # Debugging: Print paths for troubleshooting
    print(f"Base path: {base_path}")
    print(f"Relative path: {relative_path}")
    print(f"Absolute path: {absolute_path}")

    return absolute_path


# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
try:
    df = pd.read_csv(resource_path('tickets_dataset_NEW.csv'), encoding='latin1')  # Adjust file path as necessary
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
    exit(1)

# Clean text: lowercase, remove extra spaces and punctuation
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['Problem_cleaned'] = df['Problem'].apply(clean_text)

# Tokenization and removing stopwords
stop_words = set(stopwords.words('english'))
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

df['Problem_cleaned'] = df['Problem_cleaned'].apply(tokenize_and_remove_stopwords)

# Load Sentence-BERT model for embeddings
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate SBERT embeddings for all problems
print("Generating SBERT embeddings...")
embeddings = sbert_model.encode(df['Problem_cleaned'].tolist(), convert_to_tensor=True)

# Convert tensor embeddings to a numpy array
embeddings = embeddings.cpu().numpy()

# Train KNN model using SBERT embeddings
knn = NearestNeighbors(n_neighbors=3, metric='cosine')  # Adjust n_neighbors if needed
knn.fit(embeddings)

# Save the trained KNN model and SBERT model
joblib.dump(knn, 'knn_sbert_model.pkl')
joblib.dump(sbert_model, 'sbert_model.pkl')

print("SBERT model and KNN model saved successfully.")
