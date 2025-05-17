"""
Text Analytics on Shakespeare Corpus using BigQuery Public Dataset

This script demonstrates:
 1. Querying the BigQuery public Shakespeare dataset
 2. Building a term-document matrix (word counts per play)
 3. Computing TF-IDF scores for each term
 4. Extracting top TF-IDF terms for selected plays

Requirements:
  - Python 3.7+
  - pip install google-cloud-bigquery pandas scikit-learn
  - (Optional) BigQuery sandbox access or a GCP project with the BigQuery API enabled.
  - Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON if not using the sandbox.
"""
import os
from google.cloud import bigquery
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer


def query_shakespeare_table(client):
    """
    Fetch corpus, word, and word_count from the public Shakespeare table.
    """
    query = """
    SELECT
      corpus,
      word,
      word_count
    FROM
      `bigquery-public-data.samples.shakespeare`
    """
    df = client.query(query).to_dataframe()
    return df


def build_term_doc_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the raw word counts into a term-document matrix: rows=corpus, cols=word
    """
    term_doc = df.pivot(index='corpus', columns='word', values='word_count').fillna(0)
    return term_doc


def compute_tfidf(term_doc: pd.DataFrame):
    """
    Compute the TF-IDF matrix from the term-document counts.
    Returns the sparse TF-IDF matrix and the list of terms.

    TF-IDF formula with smoothing and L2 normalization.
    """
    transformer = TfidfTransformer(norm='l2', smooth_idf=True)
    tfidf = transformer.fit_transform(term_doc)
    return tfidf, term_doc.columns


def get_top_tfidf_terms(tfidf, terms, corpus_names, top_n=10):
    """
    For each corpus (row in tfidf), return the top_n terms by TF-IDF score.
    """
    top_terms = {}
    for idx, corpus in enumerate(corpus_names):
        row = tfidf[idx].toarray()[0]
        top_indices = row.argsort()[::-1][:top_n]
        top_terms[corpus] = [(terms[i], row[i]) for i in top_indices]
    return top_terms


def main():
    # Initialize BigQuery client (uses ADC or sandbox)
    client = bigquery.Client()

    print("Querying BigQuery public Shakespeare dataset...")
    df = query_shakespeare_table(client)
    print(f"Retrieved {len(df):,} records.")

    term_doc = build_term_doc_matrix(df)
    print(f"Constructed term-document matrix with shape {term_doc.shape}.")

    tfidf, terms = compute_tfidf(term_doc)
    print("Computed TF-IDF matrix.")

    # Select a few example plays
    sample_plays = ['hamlet', 'king_lear', 'romeoandjuliet']
    # Filter only those present in the index
    plays = [p for p in sample_plays if p in term_doc.index]

    print("\nTop TF-IDF terms for sample plays:")
    top = get_top_tfidf_terms(tfidf, terms, plays)
    for play in plays:
        print(f"\n=== {play.upper()} ===")
        for term, score in top[play]:
            print(f"{term}: {score:.4f}")


if __name__ == "__main__":
    main()
