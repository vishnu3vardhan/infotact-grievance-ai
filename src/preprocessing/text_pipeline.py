import os
import pandas as pd
from tqdm import tqdm
from loguru import logger

from src.preprocessing.cleaning import TextCleaner


class TextPreprocessingPipeline:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.cleaner = TextCleaner()

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_path}")
        df = pd.read_csv(self.input_path)

        if "text" not in df.columns:
            raise ValueError("Dataset must contain a 'text' column")

        logger.info(f"Dataset loaded with {len(df)} rows")
        return df

    def apply_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting text cleaning...")

        tqdm.pandas()

        df["clean_text"] = df["text"].progress_apply(
            self.cleaner.clean_text
        )

        logger.info("Basic cleaning completed")
        return df

    def apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting full preprocessing (tokenization + lemmatization)...")

        tqdm.pandas()

        df["processed_text"] = df["text"].progress_apply(
            lambda x: self.cleaner.preprocess_text(x, use_spacy=True)
        )

        logger.info("Full preprocessing completed")
        return df

    def save_data(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        df.to_csv(self.output_path, index=False)
        logger.success(f"Processed data saved to {self.output_path}")

    def run(self):
        df = self.load_data()
        df = self.apply_cleaning(df)
        df = self.apply_preprocessing(df)
        self.save_data(df)


# --------------------------
# SCRIPT ENTRY POINT
# --------------------------

if __name__ == "__main__":
    INPUT_PATH = "data/raw/civic_complaints.csv"
    OUTPUT_PATH = "data/interim/cleaned_data.csv"

    pipeline = TextPreprocessingPipeline(INPUT_PATH, OUTPUT_PATH)
    pipeline.run()