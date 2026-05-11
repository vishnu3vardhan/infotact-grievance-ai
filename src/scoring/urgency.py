"""Deterministic urgency scoring for grievance prioritization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class UrgencyPolicy:
    """Configurable urgency score policy."""

    base_scores: Mapping[str, int] = field(
        default_factory=lambda: {
            "Positive": 15,
            "Neutral": 40,
            "Negative": 70,
            "Critical/Urgent": 95,
            "Urgent": 95,
        }
    )
    confidence_adjustment: int = 20
    confidence_floor: int = -10
    confidence_ceiling: int = 10
    band_thresholds: Mapping[str, int] = field(
        default_factory=lambda: {
            "LOW": 25,
            "MEDIUM": 50,
            "HIGH": 80,
        }
    )


def _canonical_sentiment(sentiment: str) -> str:
    normalized = str(sentiment).strip()
    if normalized.lower() in {"critical", "critical/urgent", "urgent"}:
        return "Critical/Urgent"
    if normalized.lower() == "positive":
        return "Positive"
    if normalized.lower() == "neutral":
        return "Neutral"
    if normalized.lower() == "negative":
        return "Negative"
    return normalized


def priority_band_from_score(score: int, policy: UrgencyPolicy | None = None) -> str:
    """Map a 0-100 urgency score to a priority band."""
    active_policy = policy or UrgencyPolicy()
    if score >= active_policy.band_thresholds["HIGH"]:
        return "CRITICAL"
    if score >= active_policy.band_thresholds["MEDIUM"]:
        return "HIGH"
    if score >= active_policy.band_thresholds["LOW"]:
        return "MEDIUM"
    return "LOW"


def calculate_urgency_score(
    sentiment: str,
    confidence: float | None = None,
    policy: UrgencyPolicy | None = None,
) -> tuple[int, str]:
    """Convert sentiment and confidence into a bounded urgency score and band."""
    active_policy = policy or UrgencyPolicy()
    canonical_sentiment = _canonical_sentiment(sentiment)
    base_score = active_policy.base_scores.get(canonical_sentiment, 40)

    adjustment = 0
    if confidence is not None:
        centered = (float(confidence) - 0.5) * active_policy.confidence_adjustment
        adjustment = int(round(max(active_policy.confidence_floor, min(active_policy.confidence_ceiling, centered))))

    urgency_score = max(0, min(100, int(base_score + adjustment)))
    return urgency_score, priority_band_from_score(urgency_score, active_policy)