from kfp import dsl
from kfp.dsl import pipeline
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import yaml
import os

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

PROJECT_ID   = config['gcp']['project_id']
REGION       = config['gcp']['region']
BUCKET_NAME  = config['gcp']['bucket_name']
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_runs"

@pipeline(
    name="fraud-detection-pipeline",
    description="End-to-end fraud detection with champion/challenger",
    pipeline_root=PIPELINE_ROOT
)
def fraud_detection_pipeline(
    bucket_name: str = BUCKET_NAME,
    project_id: str = PROJECT_ID,
    region: str = REGION,
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
    endpoint_name: str = "fraud-detection-endpoint"
):
    from src.pipelines.components import (
        ingest_data,
        preprocess_data,
        train_model,
        champion_challenger,
        deploy_model
    )

    # Step 1: Ingest
    ingest = ingest_data(
        bucket_name=bucket_name,
        file_path="data/creditcard.csv"
    )

    # Step 2: Preprocess
    preprocess = preprocess_data(
        dataset=ingest.outputs['dataset'],
        bucket_name=bucket_name
    )

    # Step 3: Train
    train = train_model(
        processed_dataset=preprocess.outputs['processed_dataset'],
        mlflow_tracking_uri=mlflow_tracking_uri
    )

    # Step 4: Champion/Challenger
    compare = champion_challenger(
        new_model=train.outputs['model'],
        new_model_auc=train.outputs['metrics'].metadata['auc'],
        bucket_name=bucket_name,
        project_id=project_id,
        region=region,
        endpoint_name=endpoint_name
    )

    # Step 5: Deploy if challenger wins
    deploy = deploy_model(
        model=train.outputs['model'],
        deploy_decision=compare.outputs['deploy_decision'],
        project_id=project_id,
        region=region,
        bucket_name=bucket_name,
        endpoint_name=endpoint_name,
        auc=train.outputs['metrics'].metadata['auc']
    )


def run_pipeline():
    """Compile and run the pipeline"""
    import kfp.compiler as compiler

    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_detection_pipeline.yaml"
    )
    print("✅ Pipeline compiled!")

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Submit pipeline run
    job = pipeline_jobs.PipelineJob(
        display_name="fraud-detection-run",
        template_path="fraud_detection_pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=True
    )

    job.submit()
    print(f"✅ Pipeline submitted!")
    print(f"View at: https://console.cloud.google.com/vertex-ai/pipelines")


if __name__ == "__main__":
    run_pipeline()