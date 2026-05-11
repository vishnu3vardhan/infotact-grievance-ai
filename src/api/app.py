"""FastAPI application for grievance inference."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.predict import DepartmentPredictor
from src.models.predict_sentiment import SentimentPredictor
from src.scoring.urgency import calculate_urgency_score

logger = logging.getLogger(__name__)

MODEL_VERSION = os.getenv("MODEL_VERSION", "week4-1.0.0")

app = FastAPI(
    title="Infotact Grievance AI",
    version=MODEL_VERSION,
    description="Production inference API for department routing, sentiment, and urgency scoring.",
)


class PredictionRequest(BaseModel):
    """Single-text prediction payload."""

    text: str


class PredictionResponse(BaseModel):
    """Prediction response returned by the API."""

    department: str
    department_confidence: float
    sentiment: str
    sentiment_confidence: float
    urgency_score: int
    priority_band: str
    model_version: str
    timestamp: str


@lru_cache(maxsize=1)
def load_predictors() -> tuple[DepartmentPredictor, SentimentPredictor]:
    """Load and cache both inference models."""
    department_predictor = DepartmentPredictor()
    sentiment_predictor = SentimentPredictor()
    return department_predictor, sentiment_predictor


@app.on_event("startup")
def startup_event() -> None:
    """Warm the model cache during application startup."""
    try:
        load_predictors()
        logger.info("API startup complete.")
    except Exception as exc:  # pragma: no cover - startup diagnostics
        logger.exception("API startup failed: %s", exc)


@app.get("/health")
def health() -> dict[str, object]:
    """Health check endpoint used by deployment and monitoring."""
    ready = False
    message = "unhealthy"
    try:
        load_predictors()
        ready = True
        message = "healthy"
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)

    return {
        "status": message,
        "ready": ready,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict department, sentiment, and urgency from a grievance text."""
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' must be a non-empty string.")

    try:
        department_predictor, sentiment_predictor = load_predictors()
        department_prediction = department_predictor.predict([text])[0]
        department_confidence = float(department_predictor.predict_proba([text])[0].max())

        sentiment_prediction = sentiment_predictor.predict_one(text)
        urgency_score, priority_band = calculate_urgency_score(
            sentiment=sentiment_prediction.sentiment,
            confidence=sentiment_prediction.confidence,
        )

        return PredictionResponse(
            department=department_prediction,
            department_confidence=round(department_confidence, 4),
            sentiment=sentiment_prediction.sentiment,
            sentiment_confidence=round(sentiment_prediction.confidence, 4),
            urgency_score=urgency_score,
            priority_band=priority_band,
            model_version=MODEL_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except FileNotFoundError as exc:
        logger.exception("Model artifact missing: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
