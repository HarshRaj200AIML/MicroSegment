"""Data-backed unit coverage for dashboard presentation state."""

from pathlib import Path

import numpy as np

from src.ui import _cluster_distribution, _coverage, _insight_items


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cluster_distribution_keeps_noise_separate_and_uses_actual_counts():
    distribution = _cluster_distribution(np.array([0, 0, 1, -1]))

    assert distribution["label"].tolist() == ["Noise", "Cluster 1", "Cluster 2"]
    assert distribution["records"].tolist() == [1, 2, 1]
    assert distribution["percent"].tolist() == [25.0, 50.0, 25.0]
    assert distribution["color"].nunique() == 3


def test_empty_model_dashboard_state_has_no_synthetic_insights_or_coverage():
    coverage, assigned = _coverage(None, 12)
    insights = _insight_items(None, _cluster_distribution(None), None)

    assert coverage is None
    assert assigned == 0
    assert insights == [("Model not fitted", "Fit a model in Clustering Studio to generate results-backed insights.")]


def test_expanded_navigation_pages_have_concrete_streamlit_targets():
    expected_pages = [
        "8_Model_Evaluation.py",
        "9_New_Assignments.py",
        "10_Provenance.py",
        "11_Settings.py",
    ]

    for page in expected_pages:
        assert (PROJECT_ROOT / "pages" / page).is_file()
