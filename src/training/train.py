import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score,
    recall_score, f1_score,
    classification_report
)
import joblib
import mlflow
import mlflow.sklearn
import yaml
import os
from google.cloud import storage
from src.training.preprocess import preprocess, get_train_test_split, apply_smote


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def load_data(bucket_name: str = None, local_path: str = None) -> pd.DataFrame:
    """Load data from GCS or locally"""
    if local_path and os.path.exists(local_path):
        print(f"Loading from local: {local_path}")
        return pd.read_csv(local_path)

    print(f"Loading from GCS: gs://{bucket_name}/data/creditcard.csv")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob("data/creditcard.csv").download_to_filename("/tmp/creditcard.csv")
    return pd.read_csv("/tmp/creditcard.csv")


def train(df: pd.DataFrame, config: dict) -> dict:
    """Full training pipeline with MLflow tracking"""

    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run() as run:

        # Preprocess
        df, scaler = preprocess(df, fit_scaler=True)
        X_train, X_test, y_train, y_test = get_train_test_split(df)
        X_train, y_train = apply_smote(X_train, y_train)

        # Define model params
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1
        }

        # Train
        print("⏳ Training model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "auc":       round(roc_auc_score(y_test, y_prob), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "f1":        round(f1_score(y_test, y_pred), 4)
        }

        print("\n📊 Model Performance:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "fraud_model")

        # Save artifacts
        joblib.dump(model, "fraud_model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        print(f"\n✅ Run ID: {run.info.run_id}")

    return metrics, model, scaler


def evaluate_model(local_path: str = "/tmp/creditcard.csv") -> dict:
    """
    Evaluate gate — used by CI/CD pipeline
    Returns metrics dict
    """
    config = load_config()
    df = load_data(
        bucket_name=config['gcp']['bucket_name'],
        local_path=local_path
    )
    metrics, _, _ = train(df, config)
    return metrics


def save_to_gcs(bucket_name: str):
    """Upload trained model artifacts to GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for filename in ["fraud_model.pkl", "scaler.pkl"]:
        blob_path = f"models/{filename}"
        bucket.blob(blob_path).upload_from_filename(filename)
        print(f"✅ Uploaded {filename} → gs://{bucket_name}/{blob_path}")


if __name__ == "__main__":
    config = load_config()

    # Load data
    df = load_data(
        bucket_name=config['gcp']['bucket_name'],
        local_path="creditcard.csv"
    )

    # Train
    metrics, model, scaler = train(df, config)

    # Check AUC gate
    threshold = config['model']['auc_threshold']
    if metrics['auc'] < threshold:
        raise ValueError(f"❌ AUC {metrics['auc']} below threshold {threshold}")

    print(f"✅ AUC gate passed!")

    # Upload to GCS
    save_to_gcs(config['gcp']['bucket_name'])
    print("🎉 Training complete!")