# infotact-grievance-ai

AI-powered system to categorize citizen grievances, estimate sentiment, and assign urgency for Indian public-sector routing.

## Architecture

The project is organized as a small, production-oriented NLP pipeline:

- `src/preprocessing/` handles text cleaning and inference-time normalization.
- `src/models/` contains the department classifier, sentiment classifier, and predictor wrappers.
- `src/scoring/` converts sentiment plus confidence into a deterministic urgency score.
- `src/api/` exposes FastAPI inference endpoints.
- `src/evaluation/` runs final model evaluation and saves confusion matrices.

## Training And Evaluation

Run the department model training from Week 2:

```bash
python -m src.cli train_department
```

Run the sentiment model training from Week 3:

```bash
python -m src.cli train_sentiment
```

Run the final evaluation suite:

```bash
python -m src.cli evaluate_all
```

## API

Start the FastAPI service:

```bash
python -m src.cli run_api --host 0.0.0.0 --port 8000
```

Health check:

```bash
GET /health
```

Prediction endpoint:

```bash
POST /predict
Content-Type: application/json

{
	"text": "No water supply for 3 days and nobody is responding"
}
```

Example response:

```json
{
	"department": "Water",
	"department_confidence": 0.84,
	"sentiment": "Critical/Urgent",
	"sentiment_confidence": 0.91,
	"urgency_score": 96,
	"priority_band": "CRITICAL",
	"model_version": "week4-1.0.0",
	"timestamp": "2026-05-11T00:00:00+00:00"
}
```

## Artifacts

Key saved outputs:

- `artifacts/models/best_department_model.pkl`
- `artifacts/vectorizers/tfidf_vectorizer.pkl`
- `artifacts/encoders/department_label_encoder.pkl`
- `artifacts/models/best_sentiment_model.pkl`
- `artifacts/vectorizers/sentiment_tfidf_vectorizer.pkl`
- `artifacts/encoders/sentiment_label_encoder.pkl`
- `reports/department_confusion_matrix.png`
- `reports/sentiment_confusion_matrix.png`
- `reports/sentiment_model_comparison.csv`

## Notes

- The sentiment dataset contains a very small Neutral class, so the trainer performs controlled bootstrap upsampling to support stratified split and cross-validation.
- Transformer fine-tuning is wired behind a config flag in the trainer, but the default path remains the classical TF-IDF baseline for reproducibility.
