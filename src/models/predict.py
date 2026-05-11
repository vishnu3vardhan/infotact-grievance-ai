"""Prediction utilities for department classification inference."""

import os
from typing import List


def _load_pipeline(path: str):
    """Load trained pipeline from disk (joblib or pickle fallback)."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        pkl_path = path.replace(".joblib", ".pkl")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)


class DepartmentPredictor:
    """Load and serve department predictions from trained pipeline."""

    def __init__(self, pipeline_path: str = "models/department_pipeline.joblib"):
        """Initialize predictor with saved pipeline.
        
        Args:
            pipeline_path: Path to saved sklearn pipeline (TfidfVectorizer + LogisticRegression)
        """
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        self.pipeline = _load_pipeline(pipeline_path)

    def predict(self, texts: List[str]) -> List[str]:
        """Predict department for given texts.
        
        Args:
            texts: List of complaint texts
            
        Returns:
            List of predicted department names
        """
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]):
        """Get prediction probabilities for departments.
        
        Args:
            texts: List of complaint texts
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        try:
            # Try direct predict_proba on pipeline
            return self.pipeline.predict_proba(texts)
        except AttributeError:
            # Fallback: manually vectorize and predict_proba with classifier
            clf = self.pipeline.named_steps.get("clf")
            vec = self.pipeline.named_steps.get("tfidf")
            if vec is None or clf is None:
                raise RuntimeError("Pipeline does not have expected steps")
            X = vec.transform(texts)
            return clf.predict_proba(X)


if __name__ == "__main__":
    from pprint import pprint

    dp = DepartmentPredictor("models/department_pipeline.joblib")
    
    samples = [
        "There is no water supply in my street since two days",
        "Power outage in my locality since midnight",
    ]
    
    preds = dp.predict(samples)
    print("Predictions:")
    pprint(list(zip(samples, preds)))
