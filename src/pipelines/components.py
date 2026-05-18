from kfp.dsl import component, Output, Input, Dataset, Model, Metrics
from typing import NamedTuple

# ─────────────────────────────────────────
# Component 1: Data Ingestion
# ─────────────────────────────────────────
@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-storage", "pandas"]
)
def ingest_data(
    bucket_name: str,
    file_path: str,
    dataset: Output[Dataset]
):
    """Pull data from GCS bucket"""
    from google.cloud import storage
    import pandas as pd

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.download_to_filename("/tmp/data.csv")

    df = pd.read_csv("/tmp/data.csv")
    df.to_csv(dataset.path, index=False)
    print(f"✅ Ingested {len(df)} rows")


# ─────────────────────────────────────────
# Component 2: Preprocessing
# ─────────────────────────────────────────
@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib", "google-cloud-storage"]
)
def preprocess_data(
    dataset: Input[Dataset],
    bucket_name: str,
    processed_dataset: Output[Dataset]
):
    """Scale features, handle imbalance"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    from google.cloud import storage

    df = pd.read_csv(dataset.path)

    # Scale Amount and Time
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    df = df.drop(['Amount', 'Time'], axis=1)

    # Save scaler to GCS
    joblib.dump(scaler, '/tmp/scaler.pkl')
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob('models/scaler.pkl').upload_from_filename('/tmp/scaler.pkl')

    df.to_csv(processed_dataset.path, index=False)
    print(f"✅ Preprocessing done. Shape: {df.shape}")


# ─────────────────────────────────────────
# Component 3: Training
# ─────────────────────────────────────────
@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "imbalanced-learn", "joblib", "mlflow"]
)
def train_model(
    processed_dataset: Input[Dataset],
    mlflow_tracking_uri: str,
    model: Output[Model],
    metrics: Output[Metrics]
):
    """Train RandomForest with SMOTE + MLflow tracking"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, precision_score,
                                  recall_score, f1_score)
    from imblearn.over_sampling import SMOTE
    import joblib
    import mlflow

    df = pd.read_csv(processed_dataset.path)
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # MLflow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():
        # Train
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_sm, y_train_sm)

        # Evaluate
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        auc   = roc_auc_score(y_test, y_prob)
        prec  = precision_score(y_test, y_pred)
        rec   = recall_score(y_test, y_pred)
        f1    = f1_score(y_test, y_pred)

        # Log to MLflow
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 10,
            "smote": True
        })
        mlflow.log_metrics({
            "auc": auc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })
        mlflow.sklearn.log_model(clf, "fraud_model")

        print(f"✅ AUC: {auc:.4f} | F1: {f1:.4f}")

    # Save model
    joblib.dump(clf, model.path)

    # Log metrics for pipeline
    metrics.log_metric("auc", auc)
    metrics.log_metric("precision", prec)
    metrics.log_metric("recall", rec)
    metrics.log_metric("f1", f1)


# ─────────────────────────────────────────
# Component 4: Champion/Challenger
# ─────────────────────────────────────────
@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform", "joblib", "scikit-learn"]
)
def champion_challenger(
    new_model: Input[Model],
    new_model_auc: float,
    bucket_name: str,
    project_id: str,
    region: str,
    endpoint_name: str,
    deploy_decision: Output[Dataset]
):
    """Compare new model vs current champion"""
    from google.cloud import aiplatform
    import json

    aiplatform.init(project=project_id, location=region)

    # Get current champion AUC from Vertex AI
    try:
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )

        if endpoints:
            # Get champion model metadata
            endpoint = endpoints[0]
            deployed_models = endpoint.list_models()

            if deployed_models:
                champion_auc = float(
                    deployed_models[0].model.labels.get('auc', '0')
                )
                print(f"Champion AUC: {champion_auc:.4f}")
                print(f"Challenger AUC: {new_model_auc:.4f}")

                decision = "deploy" if new_model_auc > champion_auc else "keep_champion"
            else:
                decision = "deploy"  # No existing model, deploy
        else:
            decision = "deploy"  # No endpoint yet, deploy

    except Exception as e:
        print(f"No champion found: {e}")
        decision = "deploy"

    print(f"✅ Decision: {decision.upper()}")

    # Save decision
    with open(deploy_decision.path, 'w') as f:
        json.dump({
            "decision": decision,
            "new_model_auc": new_model_auc,
            "model_path": new_model.path
        }, f)


# ─────────────────────────────────────────
# Component 5: Deploy to Vertex AI
# ─────────────────────────────────────────
@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform", "joblib"]
)
def deploy_model(
    model: Input[Model],
    deploy_decision: Input[Dataset],
    project_id: str,
    region: str,
    bucket_name: str,
    endpoint_name: str,
    auc: float
):
    """Deploy to Vertex AI if challenger wins"""
    from google.cloud import aiplatform
    import json

    with open(deploy_decision.path, 'r') as f:
        decision = json.load(f)

    if decision['decision'] != 'deploy':
        print("⚠️ Champion retained. No deployment.")
        return

    aiplatform.init(project=project_id, location=region)

    # Upload model to Vertex AI
    vertex_model = aiplatform.Model.upload(
        display_name="fraud-detection-model",
        artifact_uri=f"gs://{bucket_name}/models/",
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
        ),
        labels={"auc": str(round(auc, 4))}
    )

    # Get or create endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )

    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name
        )

    # Deploy with traffic split
    vertex_model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=2,
        traffic_percentage=100
    )

    print(f"✅ Model deployed to Vertex AI!")
    print(f"Endpoint: {endpoint.resource_name}")