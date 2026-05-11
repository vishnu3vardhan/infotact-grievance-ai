import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.models.topic_modeling import run_lda
from src.models.train_department_model import train_department_classifier


def main():
    input_csv = "data/interim/cleaned_data.csv"
    report_dir = "reports/weekly"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "week2_report.md")

    # Run topic modeling
    lda, vec, topics = run_lda(input_csv, n_topics=6)

    # Train classifier (prints metrics)
    pipeline = train_department_classifier(input_csv, output_path="models/department_pipeline.joblib")

    # Load data and compute held-out metrics
    df = pd.read_csv(input_csv)
    X = df["processed_text"].fillna("")
    y = df["department"].fillna("Unknown")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use the saved pipeline to predict
    try:
        import joblib
        clf = joblib.load("models/department_pipeline.joblib")
    except Exception:
        import pickle
        with open("models/department_pipeline.pkl", "rb") as f:
            clf = pickle.load(f)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = []
    report.append(f"# Week 2 Report - Department Classification and Topic Modeling\n")
    report.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")
    report.append("## Topic Modeling (LDA)\n")
    for i, t in enumerate(topics):
        report.append(f"- Topic {i}: {t}\n")

    report.append("## Department Classifier Evaluation\n")
    report.append(f"- Test accuracy: {acc:.4f}\n")
    report.append("\n")
    report.append("### Classification report\n")
    report.append("```")
    report.append(classification_report(y_test, preds))
    report.append("```")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
