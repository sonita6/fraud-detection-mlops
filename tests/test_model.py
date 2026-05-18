import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.preprocess import preprocess, get_train_test_split, apply_smote


@pytest.fixture
def trained_model():
    """Train a quick model for testing"""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        **{f'V{i}': np.random.randn(n) for i in range(1, 29)},
        'Amount': np.random.uniform(0, 1000, n),
        'Time':   np.random.uniform(0, 172800, n),
        'Class':  np.random.choice([0, 1], n, p=[0.99, 0.01])
    })

    df, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    X_train, y_train = apply_smote(X_train, y_train)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_model_predicts(trained_model):
    model, X_test, _ = trained_model
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_model_predict_proba(trained_model):
    model, X_test, _ = trained_model
    probs = model.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert all(0 <= p <= 1 for p in probs[:, 1])


def test_model_output_binary(trained_model):
    model, X_test, _ = trained_model
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1})