#!/usr/bin/env python
"""Test the department prediction function."""

from src.models.predict import DepartmentPredictor

dp = DepartmentPredictor('models/department_pipeline.joblib')

samples = [
    'No water supply in our area for 3 days',
    'Power cuts during evening hours',
    'Garbage not collected for a week'
]

print("Testing Department Predictor\n" + "="*50)
preds = dp.predict(samples)
for text, dept in zip(samples, preds):
    print(f"Text: {text}")
    print(f"  → Department: {dept}\n")

print("="*50)
print("✓ Prediction function working!")
