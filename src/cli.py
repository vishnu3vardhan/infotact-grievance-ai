"""Simple CLI entry point for training, evaluation, and API launch."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from src.evaluation.evaluate_all import main as evaluate_all_main
from src.models.train_department_model import train_department_classifier
from src.models.train_sentiment_model import train_sentiment_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infotact grievance AI command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_department = subparsers.add_parser("train_department", help="Train the Week 2 department model")
    train_department.add_argument("--input-csv", default="data/interim/cleaned_data.csv")
    train_department.add_argument("--output-path", default="artifacts/models/best_department_model.pkl")

    train_sentiment = subparsers.add_parser("train_sentiment", help="Train the Week 3 sentiment model")
    train_sentiment.add_argument("--input-csv", default="data/interim/cleaned_data.csv")
    train_sentiment.add_argument("--model-output-path", default="artifacts/models/best_sentiment_model.pkl")
    train_sentiment.add_argument("--vectorizer-output-path", default="artifacts/vectorizers/sentiment_tfidf_vectorizer.pkl")
    train_sentiment.add_argument("--encoder-output-path", default="artifacts/encoders/sentiment_label_encoder.pkl")

    subparsers.add_parser("evaluate_all", help="Run the final evaluation suite")

    run_api = subparsers.add_parser("run_api", help="Launch the FastAPI inference service")
    run_api.add_argument("--host", default="0.0.0.0")
    run_api.add_argument("--port", default=8000, type=int)
    run_api.add_argument("--reload", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train_department":
        train_department_classifier(input_csv=args.input_csv, output_path=args.output_path)
        logger.info("Department training completed.")
        return

    if args.command == "train_sentiment":
        train_sentiment_classifier(
            input_csv=args.input_csv,
            model_output_path=args.model_output_path,
            vectorizer_output_path=args.vectorizer_output_path,
            encoder_output_path=args.encoder_output_path,
        )
        logger.info("Sentiment training completed.")
        return

    if args.command == "evaluate_all":
        evaluate_all_main()
        logger.info("Evaluation completed.")
        return

    if args.command == "run_api":
        uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)
        return


if __name__ == "__main__":
    main()
