import pandas as pd
import numpy as np
from google.cloud import bigquery
from datetime import datetime, timedelta
import yaml


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


class ModelMonitor:
    """
    Monitors model performance in production
    Tracks prediction distributions, latency,
    and business KPIs over time
    """

    def __init__(self):
        self.config    = load_config()
        self.bq_client = bigquery.Client()
        self.project   = self.config['gcp']['project_id']
        self.dataset   = self.config['bigquery']['dataset']

    def get_prediction_stats(self, hours: int = 24) -> dict:
        """Get prediction statistics for last N hours"""

        query = f"""
        SELECT
            COUNT(*)                              AS total_predictions,
            SUM(CAST(is_fraud AS INT64))          AS total_fraud_flagged,
            AVG(fraud_score)                      AS avg_fraud_score,
            MAX(fraud_score)                      AS max_fraud_score,
            MIN(fraud_score)                      AS min_fraud_score,
            COUNTIF(fraud_score >= 0.8)           AS high_risk_count,
            COUNTIF(fraud_score BETWEEN 0.5 AND 0.8) AS medium_risk_count,
            COUNTIF(fraud_score < 0.5)            AS low_risk_count
        FROM `{self.project}.{self.dataset}.fraud_results`
        WHERE timestamp >= TIMESTAMP_SUB(
            CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR
        )
        """

        results = self.bq_client.query(query).result()
        stats   = dict(list(results)[0])

        print(f"\n📊 Prediction Stats (last {hours}h):")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        return stats

    def get_fraud_rate_trend(self, days: int = 7) -> pd.DataFrame:
        """Get daily fraud rate trend"""

        query = f"""
        SELECT
            DATE(timestamp)                               AS date,
            COUNT(*)                                      AS total,
            SUM(CAST(is_fraud AS INT64))                  AS fraud_count,
            ROUND(AVG(CAST(is_fraud AS FLOAT64)) * 100, 2) AS fraud_rate_pct,
            ROUND(AVG(fraud_score), 4)                    AS avg_score
        FROM `{self.project}.{self.dataset}.fraud_results`
        WHERE timestamp >= TIMESTAMP_SUB(
            CURRENT_TIMESTAMP(), INTERVAL {days} DAY
        )
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        """

        df = self.bq_client.query(query).to_dataframe()
        print(f"\n📈 Fraud Rate Trend (last {days} days):")
        print(df.to_string(index=False))
        return df

    def check_model_health(self) -> dict:
        """
        Check if model is behaving normally
        Flags anomalies in prediction distribution
        """

        stats = self.get_prediction_stats(hours=24)

        health = {
            "timestamp":  datetime.utcnow().isoformat(),
            "status":     "healthy",
            "alerts":     []
        }

        # Alert: fraud rate too high (> 10%)
        if stats['total_predictions'] > 0:
            fraud_rate = stats['total_fraud_flagged'] / stats['total_predictions']
            if fraud_rate > 0.10:
                health["alerts"].append(
                    f"⚠️  High fraud rate: {fraud_rate:.2%}"
                )

        # Alert: avg score too high (model might be misbehaving)
        if stats['avg_fraud_score'] and stats['avg_fraud_score'] > 0.7:
            health["alerts"].append(
                f"⚠️  High avg fraud score: {stats['avg_fraud_score']:.4f}"
            )

        # Alert: no predictions in last 24h
        if stats['total_predictions'] == 0:
            health["alerts"].append(
                "⚠️  No predictions in last 24 hours!"
            )

        if health["alerts"]:
            health["status"] = "degraded"
            print(f"\n🚨 Model Health: DEGRADED")
            for alert in health["alerts"]:
                print(f"  {alert}")
        else:
            print(f"\n✅ Model Health: HEALTHY")

        return health


if __name__ == "__main__":
    monitor = ModelMonitor()
    monitor.get_prediction_stats(hours=24)
    monitor.get_fraud_rate_trend(days=7)
    monitor.check_model_health()