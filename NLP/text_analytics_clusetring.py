"""
Clustering and Topic Modeling on StackOverflow Questions using BigQuery

This script demonstrates:
 1. Querying the BigQuery public StackOverflow questions dataset by tag
 2. Cleaning and preprocessing question text (HTML removal, tokenization, stopword removal, lemmatization)
 3. Building a TF-IDF matrix and applying KMeans clustering to group similar questions
 4. Extracting top terms for each cluster
 5. Building a document-term matrix and applying Latent Dirichlet Allocation (LDA) for topic modeling
 6. Extracting top words for each topic
 7. Printing sample input-output at each step

Requirements:
  - Python 3.7+
  - pip install google-cloud-bigquery pandas scikit-learn nltk
  - nltk downloads: stopwords, wordnet
  - Set GOOGLE_APPLICATION_CREDENTIALS or use BigQuery sandbox
"""
import re
import pandas as pd
from google.cloud import bigquery
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')


def query_stackoverflow_questions(client, tag: str, limit: int = 1000) -> pd.DataFrame:
    """
    Query StackOverflow public dataset for recent questions with a given tag.
    Returns DataFrame with columns: question_id, title, body, tags

    Example:
      df = query_stackoverflow_questions(client, 'python', 500)
    """
    sql = f"""
    SELECT
      id AS question_id,
      title,
      body,
      tags
    FROM
      `bigquery-public-data.stackoverflow.posts_questions`
    WHERE
      tags LIKE '%<{tag}>%'
    LIMIT {limit}
    """
    return client.query(sql).to_dataframe()


def clean_text(text: str) -> str:
    """
    Remove HTML tags and non-alphanumeric characters, lowercase.

    Example:
      clean_text('<p>Hello <b>World!</b></p>') -> 'hello world'
    """
    # Remove HTML
    text = re.sub(r'<.*?>', ' ', text)
    # Remove non-alphanumeric
    text = re.sub(r'[^a-zA-Z0-9 ]+', ' ', text)
    # Lowercase
    text = text.lower()
    return text


def preprocess_texts(texts: pd.Series) -> pd.Series:
    """
    Tokenize, remove stopwords, and lemmatize a series of texts.

    Example:
      processed = preprocess_texts(df['clean_body'])
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def tokenize_and_lemmatize(doc):
        tokens = [w for w in doc.split() if w not in stop_words and len(w) > 2]
        lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
        return ' '.join(lemmas)

    return texts.apply(tokenize_and_lemmatize)


def cluster_questions(texts: pd.Series, n_clusters: int = 5):
    """
    Vectorize texts with TF-IDF and apply KMeans clustering.
    Returns the vectorizer, tfidf matrix, and cluster labels.

    Example:
      vect, tfidf, labels = cluster_questions(df['processed_body'], 5)
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(tfidf)
    return vectorizer, tfidf, kmeans, labels


def print_top_terms_per_cluster(vectorizer, kmeans, n_terms: int = 10):
    """
    Print top TF-IDF terms closest to each cluster centroid.
    """
    terms = vectorizer.get_feature_names_out()
    centroids = kmeans.cluster_centers_
    for i, centroid in enumerate(centroids):
        top_indices = centroid.argsort()[::-1][:n_terms]
        top_terms = [terms[idx] for idx in top_indices]
        print(f"\nCluster {i} top terms: {', '.join(top_terms)}")


def topic_modeling(texts: pd.Series, n_topics: int = 5, n_words: int = 10):
    """
    Vectorize texts with CountVectorizer and apply LDA for topic modeling.
    Print top words for each topic.

    Example:
      topic_modeling(df['processed_body'], 5, 10)
    """
    count_vect = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    dtm = count_vect.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    feature_names = count_vect.get_feature_names_out()

    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:n_words]
        top_terms = [feature_names[i] for i in top_indices]
        print(f"\nTopic {idx}: {', '.join(top_terms)}")


def main():
    client = bigquery.Client()

    # Step 1: Query Python-tagged questions
    print("Querying latest Python questions...")
    df = query_stackoverflow_questions(client, 'python', limit=500)
    print(df[['question_id', 'title', 'tags']].head())  # sample input

    # Step 2: Clean and preprocess
    df['clean_body'] = df['body'].apply(clean_text)
    df['processed_body'] = preprocess_texts(df['clean_body'])
    print("\nSample cleaned & processed text:")
    print(df['processed_body'].head())  # sample output

    # Step 3: Clustering
    print("\nClustering questions...")
    vect, tfidf, kmeans, labels = cluster_questions(df['processed_body'], n_clusters=5)
    df['cluster'] = labels
    print("Cluster distribution:")
    print(pd.Series(labels).value_counts())  # sample output
    print_top_terms_per_cluster(vect, kmeans)

    # Step 4: Topic Modeling
    print("\nTopic modeling with LDA...")
    topic_modeling(df['processed_body'], n_topics=5, n_words=10)

if __name__ == "__main__":
    main()
