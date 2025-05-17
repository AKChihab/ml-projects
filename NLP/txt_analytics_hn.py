"""
Sentiment Analysis on Hacker News Comments using BigQuery and NLTK VADER

This script demonstrates:
 1. Querying the BigQuery public Hacker News comments and stories tables
 2. Cleaning comment text (HTML tags, punctuation)
 3. Running NLTK VADER sentiment analysis on each comment
 4. Aggregating sentiment over time (monthly averages)
 5. Identifying top positive and negative stories by average comment sentiment
 6. Sample input-output demonstrations for each step

Requirements:
  - Python 3.7+
  - pip install google-cloud-bigquery pandas nltk
  - In a Python REPL or at runtime: nltk.download('vader_lexicon')
  - Set GOOGLE_APPLICATION_CREDENTIALS for service account or use BigQuery sandbox
"""
import os
import re
from datetime import datetime
import pandas as pd
from google.cloud import bigquery
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is available
nltk.download('vader_lexicon')


def query_hn_comments(client, start_date: str, end_date: str, limit: int = None) -> pd.DataFrame:
    """
    Query comments within a date range from the Hacker News public dataset.
    Returns columns: comment_id, story_id, text, comment_date

    Example:
      df = query_hn_comments(client, '2021-01-01', '2021-03-31', limit=10000)
    """
    sql = f"""
    SELECT
      id AS comment_id,
      parent AS story_id,
      COALESCE(text, '') AS text,
      DATE(TIMESTAMP_SECONDS(time)) AS comment_date
    FROM
      `bigquery-public-data.hacker_news.comments`
    WHERE
      comment_date BETWEEN '{start_date}' AND '{end_date}'
    """
    if limit:
        sql += f"\nLIMIT {limit}"
    df = client.query(sql).to_dataframe()
    return df


def clean_text(text: str) -> str:
    """
    Remove HTML tags and extra whitespace from comment text.

    Example:
      clean_text('<p>Hello <b>world</b>!</p>') -> 'Hello world!'
    """
    # Strip HTML tags
    no_html = re.sub(r'<.*?>', ' ', text)
    # Collapse whitespace
    clean = re.sub(r'\s+', ' ', no_html).strip()
    return clean


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Vader sentiment scores (compound, pos, neu, neg) for each comment.

    Example:
      df_out = analyze_sentiment(df_in)
      # df_out will have columns: compound, pos, neu, neg
    """
    sia = SentimentIntensityAnalyzer()
    # Clean text column
    df['clean_text'] = df['text'].apply(clean_text)
    # Compute sentiment scores
    scores = df['clean_text'].apply(sia.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    return pd.concat([df, scores_df], axis=1)


def aggregate_monthly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average compound sentiment per month.

    Example:
      monthly = aggregate_monthly_sentiment(df)
      # returns DataFrame with columns ['year_month', 'avg_sentiment', 'comment_count']
    """
    df['year_month'] = df['comment_date'].apply(lambda d: d.strftime('%Y-%m'))
    agg = df.groupby('year_month').agg(
        avg_sentiment=('compound', 'mean'),
        comment_count=('comment_id', 'count')
    ).reset_index()
    return agg


def top_stories_by_sentiment(df: pd.DataFrame, top_n: int = 10, positive: bool = True) -> pd.DataFrame:
    """
    Identify top stories by average comment sentiment.

    Example:
      top_pos = top_stories_by_sentiment(df, top_n=5, positive=True)
      top_neg = top_stories_by_sentiment(df, top_n=5, positive=False)
    """
    # Query story titles for mapping
    client = bigquery.Client()
    story_ids = df['story_id'].unique().tolist()
    # Flatten story_ids for SQL IN (up to a reasonable limit)
    ids_tuple = tuple(story_ids[:1000])  # limit for safety
    stories_sql = f"""
    SELECT id AS story_id, title
    FROM `bigquery-public-data.hacker_news.stories`
    WHERE id IN UNNEST({ids_tuple})
    """
    stories_df = client.query(stories_sql).to_dataframe()

    # Aggregate sentiment per story
    story_agg = df.groupby('story_id').agg(
        avg_compound=('compound', 'mean'),
        num_comments=('comment_id', 'count')
    ).reset_index()
    # Merge titles
    merged = story_agg.merge(stories_df, on='story_id', how='left')
    # Sort ascending for negatives, descending for positives
    ascending = not positive
    top = merged.sort_values('avg_compound', ascending=ascending).head(top_n)
    return top


def main():
    # Initialize BigQuery client
    client = bigquery.Client()

    # Step 1: Query comments in Q1 2021 (as an example)
    print("Querying comments from 2021-01 to 2021-03...")
    comments_df = query_hn_comments(client, '2021-01-01', '2021-03-31', limit=50000)
    print(f"Retrieved {len(comments_df):,} comments.")

    # Step 2 & 3: Clean text and analyze sentiment
    print("Analyzing sentiment...")
    sentiment_df = analyze_sentiment(comments_df)

    # Show sample input-output for sentiment analysis
    print("\n=== Sample Sentiment Output ===")
    print(sentiment_df[['text', 'clean_text', 'compound', 'pos', 'neu', 'neg']].head())

    # Step 4: Aggregate monthly sentiment
    monthly = aggregate_monthly_sentiment(sentiment_df)
    print("\n=== Monthly Average Sentiment ===")
    print(monthly)

    # Step 5: Top positive and negative stories
    top_pos = top_stories_by_sentiment(sentiment_df, top_n=5, positive=True)
    top_neg = top_stories_by_sentiment(sentiment_df, top_n=5, positive=False)

    print("\nTop 5 Most Positive Stories by Comment Sentiment:")
    print(top_pos[['title', 'avg_compound', 'num_comments']])

    print("\nTop 5 Most Negative Stories by Comment Sentiment:")
    print(top_neg[['title', 'avg_compound', 'num_comments']])

if __name__ == "__main__":
    main()
