import re

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

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

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Problem_cleaned'], df['Solution'], test_size=0.3, random_state=42)

# TF-IDF Vectorization with adjusted parameters
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize Decision Tree classifier with tuned parameters
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Use StratifiedKFold with shuffle=True to avoid n_splits error
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=cv)
clf.fit(X_train_tfidf, y_train)

print(f"Best parameters: {clf.best_params_}")

# Predict on test data
y_pred = clf.predict(X_test_tfidf)

# Evaluate the classifier
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(clf, 'decision_tree_model2.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer2.pkl')

# Example of using the loaded model for prediction
# new_text = ["New problem description to predict solution"]
# new_text_tfidf = tfidf_vectorizer.transform(new_text)
# predicted_solution = clf.predict(new_text_tfidf)
# print(f"Predicted Solution: {predicted_solution[0]}")
