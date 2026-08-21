import networkx as nx
import numpy as np

from nonlocal_apps.graph_source import (
    EtexData,
    POST_PROPAGATION_START,
    build_etex_graph,
    build_wave_model,
    cosine_dictionary_ranking,
    evaluate_etex_i,
    modal_alpha,
    pena_l1_heat_source,
)


def tiny_etex_like_data() -> EtexData:
    concentrations = np.zeros((6, 5), dtype=float)
    concentrations[0, 1] = 0.2
    concentrations[1, 2] = 1.0
    concentrations[2, 3] = 0.4
    return EtexData(
        station_ids=("A", "B", "C", "D", "E", "F"),
        latitudes=np.array([48.0, 48.2, 48.4, 48.1, 48.3, 48.5]),
        longitudes=np.array([-2.0, -1.7, -1.4, -2.4, -2.1, -1.8]),
        concentrations=concentrations,
    )


def test_graph_core_and_wave_stability_without_changing_theory():
    g = build_etex_graph(tiny_etex_like_data(), k=2)
    assert nx.is_connected(g)
    model = build_wave_model(g, max_time=4)
    lam_max = np.linalg.eigvalsh(model.laplacian)[-1]
    assert model.nu_dt**2 * lam_max < 4.0
    # The theoretical initialization remains exactly W0=W1=Id.
    assert np.allclose(model.propagators[0], np.eye(6))
    assert np.allclose(model.propagators[1], np.eye(6))
    assert modal_alpha(model, (2, 3, 4)) > 0.0


def test_cosine_rank_recovers_dictionary_column():
    g = build_etex_graph(tiny_etex_like_data(), k=2)
    model = build_wave_model(g, max_time=4)
    d = model.propagators[3]
    order, _ = cosine_dictionary_ranking(d, d[:, 2])
    assert int(order[0]) == 2


def test_etex_real_data_mapping_excludes_identity_propagators():
    data = tiny_etex_like_data()
    _, metadata = evaluate_etex_i(data, k=2, window_radius=1)
    assert min(metadata["wave_propagator_times"]) >= POST_PROPAGATION_START == 2
    assert metadata["single_wave_propagator_time"] >= 2
    assert metadata["initialization_propagators_excluded_from_real_data_matching"] == [0, 1]


def test_pena_heat_baseline_runs_on_small_graph():
    g = nx.path_graph(6)
    for u, v in g.edges:
        g[u][v]["weight"] = 1.0
    b = np.zeros(6)
    b[2] = 1.0
    estimate, x, theta, _ = pena_l1_heat_source(g, b, max_outer=3)
    assert 0 <= estimate < 6
    assert x.shape == (6,)
    assert theta > 0.0
