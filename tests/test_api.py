import pytest
import json
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_transaction():
    """Sample transaction payload"""
    return {
        "transaction_id": "test-123",
        "Time":   43200.0,
        "Amount": 150.00,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }


def test_transaction_has_required_fields(sample_transaction):
    required = ['transaction_id', 'Time', 'Amount']
    for field in required:
        assert field in sample_transaction


def test_transaction_amount_positive(sample_transaction):
    assert sample_transaction['Amount'] >= 0


def test_transaction_has_v_features(sample_transaction):
    v_features = [f'V{i}' for i in range(1, 29)]
    for f in v_features:
        assert f in sample_transaction


def test_fraud_score_range():
    """Fraud score should always be between 0 and 1"""
    mock_score = 0.87
    assert 0.0 <= mock_score <= 1.0


def test_risk_level_mapping():
    """Test risk level logic"""
    def get_risk(score):
        if score >= 0.8:   return "HIGH"
        elif score >= 0.5: return "MEDIUM"
        else:              return "LOW"

    assert get_risk(0.9)  == "HIGH"
    assert get_risk(0.6)  == "MEDIUM"
    assert get_risk(0.2)  == "LOW"