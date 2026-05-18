import json
import base64
import requests
import yaml
from google.cloud import pubsub_v1, bigquery
from datetime import datetime


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


config     = load_config()
bq_client  = bigquery.Client()


def process_transaction(message: pubsub_v1.subscriber.message.Message):
    """
    Cloud Function entry point
    Triggered when a message arrives in Pub/Sub
    """
    try:
        # Decode message
        transaction = json.loads(message.data.decode('utf-8'))
        print(f"📨 Received: {transaction.get('transaction_id')}")

        # Call fraud detection API
        cloud_run_url = "https://YOUR_CLOUD_RUN_URL/predict"

        response = requests.post(
            cloud_run_url,
            json=transaction,
            timeout=10
        )
        result = response.json()

        # Log result
        risk  = result.get('risk_level', 'UNKNOWN')
        score = result.get('fraud_score', 0)
        fraud = result.get('is_fraud', False)

        emoji = "🚨" if fraud else "✅"
        print(f"{emoji} TxID: {transaction['transaction_id']} | "
              f"Score: {score:.4f} | Risk: {risk}")

        # Acknowledge message
        message.ack()

    except Exception as e:
        print(f"❌ Error processing message: {e}")
        message.nack()  # Retry


def start_subscriber():
    """Start listening to Pub/Sub subscription"""
    project_id       = config['gcp']['project_id']
    subscription_id  = config['pubsub']['subscription']

    subscriber       = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        project_id, subscription_id
    )

    print(f"👂 Listening on: {subscription_path}")

    streaming_pull = subscriber.subscribe(
        subscription_path,
        callback=process_transaction
    )

    try:
        streaming_pull.result()
    except KeyboardInterrupt:
        streaming_pull.cancel()
        print("\n✅ Subscriber stopped")


if __name__ == "__main__":
    start_subscriber()