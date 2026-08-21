import numpy as np

from nonlocal_apps.mechanical import analyze_record, generate_synthetic_lanl_fixture


def test_mechanical_core_without_external_data():
    record = generate_synthetic_lanl_fixture(n_samples=4096, seed=7)
    result = analyze_record(record)
    assert len(result["h1_selected_hz"]) == 2
    assert np.all(np.isfinite(result["h1_selected_hz"]))
