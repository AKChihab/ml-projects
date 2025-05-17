"""
NLP Use Case: SMS Spam Detection using spaCy, GloVe Embeddings, and Spam/Ham Classification

This script demonstrates:
 1. Loading an open SMS Spam Collection dataset into a Pandas DataFrame
 2. Preprocessing text using spaCy (tokenization, lemmatization, stopword removal)
 3. Loading pre-trained GloVe embeddings and constructing average document embeddings
 4. Training a binary classifier (Logistic Regression) to distinguish Spam vs. Ham
 5. Evaluating model performance and showing sample predictions

Requirements:
  - Python 3.7+
  - pip install pandas scikit-learn spacy
  - python -m spacy download en_core_web_sm
  - Download GloVe embeddings (e.g. glove.6B.100d.txt) and set GLOVE_PATH accordingly
"""
import os
import re
import zipfile
import requests
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ----- Step 0: Configuration -----
DATA_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/"
    "data/sms.tsv"
)
GLOVE_PATH = "./glove.6B.100d.txt"  # path to downloaded GloVe file
EMBEDDING_DIM = 100

# ----- Step 1: Load SMS Spam Dataset into DataFrame -----
def load_sms_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    # Map labels to binary
    df['spam_ham'] = df['label'].map({'ham': 0, 'spam': 1})
    return df[['message', 'spam_ham']]

# ----- Step 2: Preprocess with spaCy -----
def preprocess_messages(messages: pd.Series, nlp) -> pd.Series:
    """Tokenize, lowercase, remove stopwords/punctuation, lemmatize."""
    cleaned = []
    for doc in nlp.pipe(messages, batch_size=50, disable=['ner', 'parser']):
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_stop and not token.is_punct and token.lemma_.isalpha()]
        cleaned.append(' '.join(tokens))
    return pd.Series(cleaned)

# ----- Step 3: Load GloVe Embeddings -----

def load_glove_embeddings(glove_path: str) -> dict:
    """Read GloVe file into a dictionary: word -> numpy array."""
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vec = np.array(parts[1:], dtype='float32')
            embeddings[word] = vec
    return embeddings

# ----- Step 4: Vectorize via Average GloVe Embeddings -----
def embed_messages(texts: pd.Series, embeddings: dict, dim: int) -> np.ndarray:
    """Compute average embedding for each text."""
    X = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        words = text.split()
        # collect vectors for in-vocab words
        vecs = [embeddings[w] for w in words if w in embeddings]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
        else:
            X[i] = np.zeros(dim)
    return X

# ----- Main Pipeline -----
def main():
    # Load data
    df = load_sms_data(DATA_URL)
    print(f"Loaded {len(df)} messages (spam: {df['spam_ham'].sum()}, ham: {len(df) - df['spam_ham'].sum()})")

    # Initialize spaCy
    nlp = spacy.load('en_core_web_sm')

    # Preprocess text
    print("Preprocessing messages with spaCy...")
    df['clean_msg'] = preprocess_messages(df['message'], nlp)
    print("Sample cleaned messages:")
    print(df[['message', 'clean_msg', 'spam_ham']].head(), '\n')

    # Load embeddings
    print("Loading GloVe embeddings...")
    if not os.path.exists(GLOVE_PATH):
        raise FileNotFoundError(f"GloVe file not found. Please download to {GLOVE_PATH}")
    glove_embeddings = load_glove_embeddings(GLOVE_PATH)

    # Embed messages
    print("Computing message embeddings...")
    X = embed_messages(df['clean_msg'], glove_embeddings, EMBEDDING_DIM)
    y = df['spam_ham'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)

    # Train classifier
    print("Training Logistic Regression classifier for Spam/Ham...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Ham','Spam']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

    # Sample predictions
    sample_idx = np.random.choice(len(X_test), size=5, replace=False)
    print("=== Sample Predictions ===")
    for idx in sample_idx:
        print(f"Message: {df.iloc[X_test.tolist().index(X_test[idx].tolist())]['message']}")
        print(f"Cleaned: {df.iloc[X_test.tolist().index(X_test[idx].tolist())]['clean_msg']}")
        print(f"Actual: {'Spam' if y_test[idx]==1 else 'Ham'}, Predicted: {'Spam' if y_pred[idx]==1 else 'Ham'}\n")

if __name__ == '__main__':
    main()
