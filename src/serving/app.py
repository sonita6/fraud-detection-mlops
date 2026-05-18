from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from google.cloud import storage, bigquery
import yaml
import os
import uuid
from datetime import datetime

app = Flask(__name__)

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Load model + scaler from GCS on startup
def load_artifacts():
    client = storage.Client()
    bucket = client.bucket(config['gcp']['bucket_name'])

    bucket.blob("models/fraud_model.pkl").download_to_filename("/tmp/fraud_model.pkl")
    bucket.blob("models/scaler.pkl").download_to_filename("/tmp/scaler.pkl")

    model  = joblib.load("/tmp/fraud_model.pkl")
    scaler = joblib.load("/tmp/scaler.pkl")

    print("✅ Model and scaler loaded!")
    return model, scaler

model, scaler = load_artifacts()
bq_client = bigquery.Client()


def log_to_bigquery(transaction_id, amount, time, fraud_score, is_fraud):
    """Log every prediction to BigQuery"""
    table_id = f"{config['gcp']['project_id']}.{config['bigquery']['dataset']}.{config['bigquery']['table']}"

    rows = [{
        "transaction_id": transaction_id,
        "time":           float(time),
        "amount":         float(amount),
        "fraud_score":    float(fraud_score),
        "is_fraud":       bool(is_fraud),
        "timestamp":      datetime.utcnow().isoformat()
    }]

    errors = bq_client.insert_rows_json(table_id, rows)
    if errors:
        print(f"BigQuery error: {errors}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "fraud-detection-v1"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build feature dataframe
        features = pd.DataFrame([{
            k: v for k, v in data.items()
            if k not in ['transaction_id']
        }])

        # Scale Amount and Time
        features['Amount_scaled'] = scaler.transform(features[['Amount']])
        features['Time_scaled']   = scaler.transform(features[['Time']])
        features = features.drop(['Amount', 'Time'], axis=1)

        # Predict
        fraud_score = float(model.predict_proba(features)[0][1])
        is_fraud    = fraud_score >= config['serving']['threshold']
        risk_level  = (
            "HIGH"   if fraud_score >= 0.8 else
            "MEDIUM" if fraud_score >= 0.5 else
            "LOW"
        )

        transaction_id = data.get('transaction_id', str(uuid.uuid4()))

        # Log to BigQuery
        log_to_bigquery(
            transaction_id,
            data.get('Amount', 0),
            data.get('Time', 0),
            fraud_score,
            is_fraud
        )

        return jsonify({
            "transaction_id": transaction_id,
            "is_fraud":       is_fraud,
            "fraud_score":    round(fraud_score, 4),
            "risk_level":     risk_level,
            "timestamp":      datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", config['serving']['port']))
    app.run(host="0.0.0.0", port=port, debug=False)