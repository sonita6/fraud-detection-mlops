import pandas as pd
import numpy as np
from scipy import stats
from google.cloud import bigquery
from datetime import datetime
import yaml


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


class DriftDetector:
    """
    Detects feature drift between training data
    and incoming live transactions using KS Test
    """

    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        self.config = load_config()
        self.bq_client = bigquery.Client()

    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        """Run KS test on each feature"""

        drift_report = {
            "timestamp":          datetime.utcnow().isoformat(),
            "features_drifted":   [],
            "features_stable":    [],
            "drift_detected":     False,
            "drift_score":        0.0,
            "details":            {}
        }

        for column in self.reference_data.columns:
            if column == 'Class':
                continue

            # Kolmogorov-Smirnov Test
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[column].dropna(),
                current_data[column].dropna()
            )

            drifted = p_value < self.threshold

            drift_report["details"][column] = {
                "ks_statistic": round(ks_stat, 4),
                "p_value":      round(p_value, 4),
                "drifted":      drifted
            }

            if drifted:
                drift_report["features_drifted"].append(column)
            else:
                drift_report["features_stable"].append(column)

        # Summary
        total = len(drift_report["details"])
        drifted = len(drift_report["features_drifted"])

        drift_report["drift_detected"] = drifted > 0
        drift_report["drift_score"]    = round(drifted / total, 4) if total > 0 else 0.0

        # Print summary
        print(f"\n📊 Drift Report:")
        print(f"  Total features:    {total}")
        print(f"  Drifted features:  {drifted}")
        print(f"  Drift score:       {drift_report['drift_score']}")
        print(f"  Drift detected:    {drift_report['drift_detected']}")

        if drift_report["features_drifted"]:
            print(f"  Drifted features: {drift_report['features_drifted']}")

        return drift_report

    def log_to_bigquery(self, drift_report: dict):
        """Log drift results to BigQuery"""
        project_id = self.config['gcp']['project_id']
        dataset    = self.config['bigquery']['dataset']
        table      = self.config['bigquery']['drift_table']
        table_id   = f"{project_id}.{dataset}.{table}"

        rows = [{
            "timestamp":            drift_report["timestamp"],
            "drift_detected":       drift_report["drift_detected"],
            "drift_score":          drift_report["drift_score"],
            "features_drifted":     str(drift_report["features_drifted"]),
            "num_features_drifted": len(drift_report["features_drifted"]),
            "total_features":       len(drift_report["details"])
        }]

        errors = self.bq_client.insert_rows_json(table_id, rows)

        if not errors:
            print(f"✅ Drift report logged to BigQuery")
        else:
            print(f"❌ BigQuery error: {errors}")

        return drift_report

    def should_retrain(self, drift_report: dict, threshold: float = 0.3) -> bool:
        """
        Trigger retraining if drift score exceeds threshold
        e.g. 30% of features have drifted
        """
        should = drift_report["drift_score"] >= threshold
        if should:
            print(f"⚠️  Drift score {drift_report['drift_score']} >= {threshold}")
            print(f"🔄 Retraining recommended!")
        else:
            print(f"✅ Drift score {drift_report['drift_score']} < {threshold}")
            print(f"✅ No retraining needed")
        return should


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    import numpy as np

    # Simulate reference and current data
    reference = pd.DataFrame(np.random.randn(1000, 5),
                              columns=['V1','V2','V3','Amount','Time'])
    current   = pd.DataFrame(np.random.randn(100, 5) + 0.5,
                              columns=['V1','V2','V3','Amount','Time'])

    detector = DriftDetector(reference_data=reference)
    report   = detector.detect_drift(current)
    print(f"\nShould retrain: {detector.should_retrain(report)}")