import numpy as np

from nonlocal_apps.mechanical import (
    analyze_rwth_record,
    generate_synthetic_rwth_fixture,
    group_velocity,
    isolation_required_k0,
    manuscript_lattice_numbers,
    nn_nnn_band_edge,
)


def test_rwth_core_without_external_data():
    record = generate_synthetic_rwth_fixture(n_samples=12000, seed=7)
    result = analyze_rwth_record(record)
    summary = result["summary"]
    assert 3.0 < summary["dominant_response_frequency_hz"] < 3.5
    assert summary["acceleration_transfer_norm"] > 0
    assert summary["displacement_transfer_norm"] > 0
    assert 0.0 <= summary["transmissibility_coherence"] <= 1.0


def test_manuscript_nn_nnn_band_values():
    expected = {
        0.10: (4.3600, 2.0881, None),
        0.25: (4.3600, 2.0881, None),
        0.40: (4.5850, 2.1413, 2.2459),
        0.50: (4.8600, 2.2045, 2.0944),
        0.70: (5.5171, 2.3489, 1.9360),
    }
    for zeta, (lam, omg, theta) in expected.items():
        row = nn_nnn_band_edge(zeta)
        assert abs(row["lambda_max"] - lam) < 6e-5
        assert abs(row["omega_max"] - omg) < 6e-5
        if theta is None:
            assert row["theta_star"] is None
        else:
            assert abs(row["theta_star"] - theta) < 6e-5


def test_manuscript_group_velocity_and_isolation_numbers():
    assert abs(group_velocity(2.5, 0.10) - 0.2007) < 6e-5
    assert abs(group_velocity(2.5, 0.50) - (-0.1666)) < 6e-5
    assert abs(isolation_required_k0(0.35, 0.50) - 0.7484) < 6e-5
    values = manuscript_lattice_numbers()
    assert abs(values["max_gains"]["k0_0.36"] - 8.5436) < 8e-4
    assert abs(values["max_gains"]["k0_0.80"] - 1.8134) < 8e-4
