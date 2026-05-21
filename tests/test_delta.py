import numpy as np

from asne.delta import compute_delta, summarize_delta


def test_compute_delta_subtracts_baseline_from_steered():
    baseline = {"response": np.array([1.0, 2.0, 3.0])}
    steered = {"response": np.array([1.5, 1.0, 5.0])}

    delta = compute_delta(baseline, steered)

    np.testing.assert_allclose(delta, np.array([0.5, -1.0, 2.0]))


def test_summarize_delta_returns_expected_fields():
    summary = summarize_delta(np.array([0.5, -3.0, 1.0]), top_k=2)

    assert summary["mean_abs_delta"] == 1.5
    assert summary["max_abs_delta"] == 3.0
    assert summary["top_indices"] == [1, 2]
    assert summary["top_values"] == [-3.0, 1.0]
