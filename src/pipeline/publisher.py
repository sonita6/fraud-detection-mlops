import pandas as pd
import json
import time
import uuid
import yaml
from google.cloud import pubsub_v1
from datetime import datetime


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def simulate_live_transactions(
    csv_path: str,
    delay: float = 1.0,
    max_transactions: int = None
):
    """
    Simulate live transaction stream
    by publishing CSV rows to Pub/Sub
    """
    config     = load_config()
    project_id = config['gcp']['project_id']
    topic_id   = config['pubsub']['topic']

    publisher  = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} transactions")
    print(f"📡 Publishing to: {topic_path}")
    print(f"⏱️  Delay between transactions: {delay}s\n")

    published = 0

    for idx, row in df.iterrows():
        # Build transaction message
        transaction = row.to_dict()
        transaction['transaction_id'] = str(uuid.uuid4())
        transaction['published_at']   = datetime.utcnow().isoformat()

        # Publish to Pub/Sub
        message = json.dumps(transaction).encode('utf-8')
        future  = publisher.publish(topic_path, message)
        msg_id  = future.result()

        published += 1
        fraud_label = "🚨 FRAUD" if row['Class'] == 1 else "✅ NORMAL"
        print(f"[{published}] {fraud_label} | Amount: ${row['Amount']:.2f} | MsgID: {msg_id}")

        # Stop if max reached
        if max_transactions and published >= max_transactions:
            print(f"\n✅ Published {published} transactions")
            break

        time.sleep(delay)

    print(f"\n🎉 Done! Published {published} transactions")


if __name__ == "__main__":
    simulate_live_transactions(
        csv_path="creditcard.csv",
        delay=1.0,
        max_transactions=100  # Start with 100 for testing
    )