import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import yaml
import os


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def preprocess(df: pd.DataFrame, fit_scaler: bool = True, scaler=None):
    """
    Preprocess raw credit card data.
    fit_scaler=True during training, False during serving
    """

    # Scale Amount and Time
    if fit_scaler:
        scaler = StandardScaler()
        df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
        df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    else:
        # Use existing scaler during serving
        df['Amount_scaled'] = scaler.transform(df[['Amount']])
        df['Time_scaled']   = scaler.transform(df[['Time']])

    # Drop original columns
    df = df.drop(['Amount', 'Time'], axis=1)

    return df, scaler


def get_train_test_split(df: pd.DataFrame):
    """Split into train/test with stratification"""
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Handle class imbalance with SMOTE"""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"After SMOTE:  {pd.Series(y_resampled).value_counts().to_dict()}")

    return X_resampled, y_resampled


if __name__ == "__main__":
    # Quick test
    df = pd.read_csv("creditcard.csv")
    df, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    X_train, y_train = apply_smote(X_train, y_train)
    print(f"✅ Preprocessing done!")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")