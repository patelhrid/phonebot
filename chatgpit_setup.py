import subprocess
import logging
import re
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import joblib
import sys
import os


def tokenize_and_remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def setup_once():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    base_path = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_path, "cache_dir")

    # Loading pre-trained model
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Load dataset
    df = pd.read_csv(resource_path('tickets_dataset_NEW.csv'), encoding='latin1')

    # Data preprocessing
    df['Problem_cleaned'] = df['Problem'].apply(clean_text)
    df['Problem_cleaned'] = df['Problem_cleaned'].apply(tokenize_and_remove_stopwords)

    embeddings = sbert_model.encode(df['Problem_cleaned'].tolist(), convert_to_tensor=True).cpu().numpy()

    # Train KNN model
    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(embeddings)

    # Save models
    joblib.dump(knn, 'knn_sbert_model.pkl')
    joblib.dump(sbert_model, 'sbert_model.pkl')

    logger.info("Model setup completed successfully.")


if __name__ == "__main__":
    setup_once()
