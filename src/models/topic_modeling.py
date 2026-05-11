"""
Topic Modeling Module for Indian Civic Complaints.

This module implements Latent Dirichlet Allocation (LDA) for discovering topics
in citizen grievance texts. It provides a complete pipeline from data loading
to model training, topic extraction, and artifact saving.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_PATH = Path("data/interim/cleaned_data.csv")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/topics")
DEFAULT_LDA_MODEL_PATH = DEFAULT_ARTIFACTS_DIR / "lda_model.pkl"
DEFAULT_VECTORIZER_PATH = DEFAULT_ARTIFACTS_DIR / "count_vectorizer.pkl"


def load_dataset(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned dataset containing processed text.

    Args:
        data_path: Path to the CSV file with cleaned data.

    Returns:
        DataFrame with complaint data.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    if "processed_text" not in df.columns:
        raise ValueError("Dataset must contain 'processed_text' column")

    logger.info(f"Loaded {len(df)} complaints")
    return df


def vectorize_text(
    corpus: List[str],
    max_features: int = 2000,
    stop_words: str = "english"
) -> Tuple[CountVectorizer, np.ndarray]:
    """
    Vectorize text corpus using CountVectorizer.

    Args:
        corpus: List of processed text strings.
        max_features: Maximum number of features (vocabulary size).
        stop_words: Stop words to remove.

    Returns:
        Tuple of (fitted vectorizer, document-term matrix).
    """
    logger.info(f"Vectorizing {len(corpus)} documents with max_features={max_features}")

    vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
    dtm = vectorizer.fit_transform(corpus)

    logger.info(f"Created document-term matrix: {dtm.shape}")
    return vectorizer, dtm


def train_lda_model(
    dtm: np.ndarray,
    n_topics: int = 5,
    random_state: int = 42,
    max_iter: int = 10
) -> LatentDirichletAllocation:
    """
    Train LDA model on document-term matrix.

    Args:
        dtm: Document-term matrix from vectorization.
        n_topics: Number of topics to discover.
        random_state: Random seed for reproducibility.
        max_iter: Maximum iterations for LDA.

    Returns:
        Trained LDA model.
    """
    logger.info(f"Training LDA model with {n_topics} topics")

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=max_iter,
        learning_method="online",
        learning_offset=50.0
    )

    lda.fit(dtm)
    logger.info("LDA training completed")
    return lda


def extract_topics(
    lda_model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    n_top_words: int = 10
) -> List[List[str]]:
    """
    Extract top words for each topic.

    Args:
        lda_model: Trained LDA model.
        vectorizer: Fitted CountVectorizer.
        n_top_words: Number of top words per topic.

    Returns:
        List of lists, each containing top words for a topic.
    """
    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics.append(top_features)

    logger.info(f"Extracted {len(topics)} topics with {n_top_words} words each")
    return topics


def display_topics(topics: List[List[str]]) -> None:
    """
    Display topics in a readable format.

    Args:
        topics: List of topic word lists.
    """
    print("\n" + "="*50)
    print("DISCOVERED TOPICS")
    print("="*50)

    for i, topic_words in enumerate(topics, 1):
        words_str = ", ".join(topic_words)
        print(f"Topic {i}:")
        print(f"{words_str}")
        print()


def save_topic_model(
    lda_model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    model_path: Path = DEFAULT_LDA_MODEL_PATH,
    vectorizer_path: Path = DEFAULT_VECTORIZER_PATH
) -> None:
    """
    Save LDA model and vectorizer to disk.

    Args:
        lda_model: Trained LDA model.
        vectorizer: Fitted CountVectorizer.
        model_path: Path to save LDA model.
        vectorizer_path: Path to save vectorizer.
    """
    DEFAULT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving LDA model to {model_path}")
    joblib.dump(lda_model, model_path)

    logger.info(f"Saving vectorizer to {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)

    logger.info("Topic modeling artifacts saved successfully")


def run_topic_modeling_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    n_topics: int = 5,
    max_features: int = 2000,
    n_top_words: int = 10,
    save_artifacts: bool = True
) -> Tuple[LatentDirichletAllocation, CountVectorizer, List[List[str]]]:
    """
    Complete topic modeling pipeline.

    Args:
        data_path: Path to cleaned dataset.
        n_topics: Number of topics to discover.
        max_features: Max features for vectorization.
        n_top_words: Top words per topic to display.
        save_artifacts: Whether to save model and vectorizer.

    Returns:
        Tuple of (LDA model, vectorizer, topics as word lists).
    """
    logger.info("Starting topic modeling pipeline")

    # Load data
    df = load_dataset(data_path)
    corpus = df["processed_text"].fillna("").tolist()

    # Vectorize
    vectorizer, dtm = vectorize_text(corpus, max_features=max_features)

    # Train LDA
    lda_model = train_lda_model(dtm, n_topics=n_topics)

    # Extract topics
    topics = extract_topics(lda_model, vectorizer, n_top_words=n_top_words)

    # Display
    display_topics(topics)

    # Save
    if save_artifacts:
        save_topic_model(lda_model, vectorizer)

    logger.info("Topic modeling pipeline completed")
    return lda_model, vectorizer, topics


if __name__ == "__main__":
    # Run with defaults
    run_topic_modeling_pipeline()
