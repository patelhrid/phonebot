import re

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
try:
    df = pd.read_csv('../tickets_dataset.csv', encoding='latin1')  # Adjust file path as necessary
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

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(df['Problem_cleaned'])

# Train KNN model
knn = NearestNeighbors(n_neighbors=1, metric='cosine')
knn.fit(X_tfidf)

# Save model and vectorizer
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")
