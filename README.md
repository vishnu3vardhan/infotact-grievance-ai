# 🚀 Infotact Grievance AI

> **AI-powered grievance intelligence system for Indian public-sector complaint routing.**
>
> Automatically classifies citizen complaints into departments, detects sentiment severity, and computes an urgency score to help government agencies prioritize and route complaints faster.

---

## 📌 Overview

**Infotact Grievance AI** is a production-oriented Natural Language Processing (NLP) system that analyzes citizen grievances submitted through public-service portals.

Given a complaint such as:

> **"No water supply for 3 days and nobody is responding."**

The system automatically:

* 🏢 Predicts the responsible department (e.g., Water, Electricity, Roads)
* 💬 Detects sentiment severity (Critical/Urgent, Negative, Neutral, Positive)
* 🚨 Computes an urgency score (0–100)
* 🏷️ Assigns a priority band (LOW, MEDIUM, HIGH, CRITICAL)
* 🌐 Returns results via a FastAPI REST API

---

## ✨ Key Features

* **Department Classification** using TF-IDF + classical ML
* **Sentiment Severity Detection** tailored for grievance language
* **Deterministic Urgency Scoring Engine** combining sentiment and confidence
* **FastAPI Inference Service** with production-ready JSON responses
* **CLI-Based Training and Evaluation Workflow**
* **Automated Reports** including confusion matrices and model comparisons
* **Reproducible Artifact Management** for models, vectorizers, and encoders

---

## 🏗️ Project Architecture

```text
infotact-grievance-ai/
│
├── src/
│   ├── preprocessing/      # Text cleaning and normalization
│   ├── models/             # Department and sentiment models
│   ├── scoring/            # Urgency score calculation
│   ├── api/                # FastAPI routes and schemas
│   ├── evaluation/         # Metrics and confusion matrices
│   └── cli.py              # Unified command-line interface
│
├── artifacts/
│   ├── models/
│   ├── vectorizers/
│   └── encoders/
│
├── reports/
│   ├── department_confusion_matrix.png
│   ├── sentiment_confusion_matrix.png
│   └── sentiment_model_comparison.csv
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer         | Technology                 |
| ------------- | -------------------------- |
| Language      | Python 3.10+               |
| ML            | scikit-learn               |
| NLP           | TF-IDF Vectorization       |
| API           | FastAPI                    |
| Validation    | Pydantic                   |
| Visualization | Matplotlib, Seaborn        |
| Serialization | joblib                     |
| CLI           | argparse / Typer-style CLI |

---

## 🔄 End-to-End Workflow

```text
Citizen Complaint
      ↓
Text Preprocessing
      ↓
Department Classifier
      ↓
Sentiment Classifier
      ↓
Urgency Scoring Engine
      ↓
Priority Band Assignment
      ↓
FastAPI JSON Response
```

---

## 📂 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/infotact-grievance-ai.git
cd infotact-grievance-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

### Train Department Classifier

```bash
python -m src.cli train_department
```

### Train Sentiment Classifier

```bash
python -m src.cli train_sentiment
```

---

## 📊 Evaluation

Run the full evaluation suite:

```bash
python -m src.cli evaluate_all
```

Generated reports:

* `reports/department_confusion_matrix.png`
* `reports/sentiment_confusion_matrix.png`
* `reports/sentiment_model_comparison.csv`

---

## 🌐 API Usage

### Start the API Server

```bash
python -m src.cli run_api --host 0.0.0.0 --port 8000
```

The API will be available at:

* Swagger Docs: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

---

## ❤️ Health Check

### Request

```http
GET /health
```

### Response

```json
{
  "status": "ok",
  "model_version": "week4-1.0.0"
}
```

---

## 🔮 Prediction Endpoint

### Request

```http
POST /predict
Content-Type: application/json
```

```json
{
  "text": "No water supply for 3 days and nobody is responding"
}
```

### Response

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

---

## 🚨 Urgency Scoring Logic

The urgency score combines:

* Sentiment severity
* Prediction confidence
* Rule-based weighting

### Example Mapping

| Sentiment       | Typical Score Range | Priority Band |
| --------------- | ------------------- | ------------- |
| Positive        | 0–20                | LOW           |
| Neutral         | 20–40               | LOW / MEDIUM  |
| Negative        | 40–80               | HIGH          |
| Critical/Urgent | 80–100              | CRITICAL      |

This deterministic approach ensures explainable prioritization.

---

## 📁 Saved Artifacts

### Models

* `artifacts/models/best_department_model.pkl`
* `artifacts/models/best_sentiment_model.pkl`

### Vectorizers

* `artifacts/vectorizers/tfidf_vectorizer.pkl`
* `artifacts/vectorizers/sentiment_tfidf_vectorizer.pkl`

### Label Encoders

* `artifacts/encoders/department_label_encoder.pkl`
* `artifacts/encoders/sentiment_label_encoder.pkl`

---

## 🧪 Example Python Usage

```python
from src.models.predictor import GrievancePredictor

predictor = GrievancePredictor()

result = predictor.predict(
    "Street lights are not working in my colony for two nights"
)

print(result)
```

---

## 📈 Model Design Choices

### Department Classification

* TF-IDF Vectorization
* Logistic Regression / Linear SVM
* Multi-class classification

### Sentiment Classification

* Custom labels for grievance severity
* Bootstrap upsampling for minority Neutral class

### Transformer Support

Transformer fine-tuning is wired behind a configuration flag, but classical TF-IDF models remain the default path for reproducibility and faster training.

---

## 📝 Special Notes

* The sentiment dataset contains a very small **Neutral** class.
* Controlled bootstrap upsampling is used to support stratified splitting and cross-validation.
* Confidence scores are incorporated into urgency estimation.
* The pipeline is modular and production-friendly.

---

## 🛣️ Future Enhancements

* Multilingual grievance support (Hindi, Telugu, Tamil)
* Named Entity Recognition (locations and departments)
* Complaint deduplication
* Auto-ticket assignment
* Docker deployment
* CI/CD pipelines
* Monitoring and drift detection

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Your Name**

Built as an AI-driven solution for smarter citizen grievance routing and prioritization.



---

> **Infotact Grievance AI** transforms unstructured citizen complaints into actionable intelligence for public-sector response systems.
