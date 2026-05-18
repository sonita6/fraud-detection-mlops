import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.preprocess import preprocess, get_train_test_split, apply_smote


@pytest.fixture
def sample_df():
    """Create sample dataframe for testing"""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        **{f'V{i}': np.random.randn(n) for i in range(1, 29)},
        'Amount': np.random.uniform(0, 1000, n),
        'Time':   np.random.uniform(0, 172800, n),
        'Class':  np.random.choice([0, 1], n, p=[0.998, 0.002])
    })
    return df


def test_preprocess_creates_scaled_columns(sample_df):
    df, scaler = preprocess(sample_df.copy())
    assert 'Amount_scaled' in df.columns
    assert 'Time_scaled' in df.columns


def test_preprocess_drops_original_columns(sample_df):
    df, scaler = preprocess(sample_df.copy())
    assert 'Amount' not in df.columns
    assert 'Time' not in df.columns


def test_preprocess_no_nulls(sample_df):
    df, scaler = preprocess(sample_df.copy())
    assert df.isnull().sum().sum() == 0


def test_train_test_split_ratio(sample_df):
    df, _ = preprocess(sample_df.copy())
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.01


def test_smote_balances_classes(sample_df):
    df, _ = preprocess(sample_df.copy())
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    X_res, y_res = apply_smote(X_train, y_train)
    counts = pd.Series(y_res).value_counts()
    assert counts[0] == counts[1]