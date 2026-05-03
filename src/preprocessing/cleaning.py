import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
from loguru import logger


# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise


class TextCleaner:
    """
    A reusable text cleaning and preprocessing class for Indian civic complaints.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.custom_stopwords = {
            "sir", "madam", "please", "kindly", "regards", "thank",
            "thanks", "dear", "request"
        }
        self.stop_words.update(self.custom_stopwords)

        self.lemmatizer = WordNetLemmatizer()

        # Common Hinglish normalization (extend later if needed)
        self.hinglish_dict = {
            "paani": "water",
            "bijli": "electricity",
            "sadak": "road",
            "kachra": "garbage",
            "ganda": "dirty"
        }

    # --------------------------
    # BASIC CLEANING FUNCTIONS
    # --------------------------

    def to_lower(self, text: str) -> str:
        return text.lower()

    def remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\S+", "", text)

    def remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+", "", text)

    def remove_phone_numbers(self, text: str) -> str:
        return re.sub(r"\b\d{10}\b", "", text)

    def remove_special_characters(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_numbers(self, text: str) -> str:
        return re.sub(r"\d+", "", text)

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def remove_repeated_chars(self, text: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    # --------------------------
    # INDIA-SPECIFIC CLEANING
    # --------------------------

    def normalize_hinglish(self, text: str) -> str:
        words = text.split()
        normalized = [self.hinglish_dict.get(word, word) for word in words]
        return " ".join(normalized)

    def normalize_abbreviations(self, text: str) -> str:
        abbreviations = {
            "govt": "government",
            "pls": "please",
            "plz": "please",
            "dept": "department",
            "elec": "electricity"
        }

        words = text.split()
        normalized = [abbreviations.get(word, word) for word in words]
        return " ".join(normalized)

    # --------------------------
    # NLP PROCESSING
    # --------------------------

    def tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def spacy_lemmatize(self, text: str) -> List[str]:
        doc = nlp(text)
        return [token.lemma_ for token in doc if token.text not in self.stop_words]

    # --------------------------
    # FULL PIPELINE
    # --------------------------

    def clean_text(self, text: str) -> str:
        """
        Full cleaning pipeline (no tokenization)
        """
        text = self.to_lower(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_phone_numbers(text)
        text = self.normalize_abbreviations(text)
        text = self.normalize_hinglish(text)
        text = self.remove_repeated_chars(text)
        text = self.remove_special_characters(text)
        text = self.remove_numbers(text)
        text = self.normalize_whitespace(text)

        return text

    def preprocess_text(self, text: str, use_spacy: bool = True) -> str:
        """
        Full preprocessing pipeline:
        cleaning + tokenization + stopword removal + lemmatization
        """
        cleaned = self.clean_text(text)

        if use_spacy:
            tokens = self.spacy_lemmatize(cleaned)
        else:
            tokens = self.tokenize(cleaned)
            tokens = self.remove_stopwords(tokens)
            tokens = self.lemmatize(tokens)

        return " ".join(tokens)


# --------------------------
# QUICK TEST (optional)
# --------------------------

if __name__ == "__main__":
    cleaner = TextCleaner()

    sample_text = """
    Sir, there is no water supply in our area since 3 days!!!
    Pls help urgently. Road is also damaged.
    Visit http://example.com for details.
    """

    print("Original:\n", sample_text)
    print("\nCleaned:\n", cleaner.clean_text(sample_text))
    print("\nProcessed:\n", cleaner.preprocess_text(sample_text))