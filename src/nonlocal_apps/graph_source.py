from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ETEX_SOURCE_LAT = 48.0583
ETEX_SOURCE_LON = -2.0083


@dataclass(frozen=True)
class WaveModel:
    graph: nx.Graph
    laplacian: np.ndarray
    nu_dt: float
    propagators: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class EtexData:
    station_ids: tuple[str, ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    concentrations: np.ndarray  # rows=stations, columns=3-hour snapshots

    @property
    def n_stations(self) -> int:
        return len(self.station_ids)

    @property
    def n_times(self) -> int:
        return int(self.concentrations.shape[1])


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", text)]


def _clean_station_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", token).upper()


def load_etex_station_list(path: str | Path) -> pd.DataFrame:
    """Parse the JRC ``stationlist.950130`` file.

    The historical ETEX text files are fixed/whitespace formatted. This parser intentionally
    avoids depending on a single column layout: it finds a station identifier plus the first
    physically plausible latitude/longitude pair on each line.
    """
    path = Path(path)
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "!", "%")):
            continue
        tokens = line.split()
        if not tokens:
            continue
        station = _clean_station_token(tokens[0])
        if not station or station.isdigit():
            continue
        nums = _numbers(line)
        if len(nums) < 2:
            continue
        found = None
        # Prefer decimal-degree pairs: Europe is roughly lat 30..75, lon -30..50.
        for i in range(len(nums) - 1):
            a, b = nums[i], nums[i + 1]
            if 30.0 <= a <= 75.0 and -30.0 <= b <= 50.0:
                found = (a, b)
                break
            if -30.0 <= a <= 50.0 and 30.0 <= b <= 75.0:
                found = (b, a)
                break
        if found is None:
            continue
        lat, lon = found
        rows.append({"station_id": station, "latitude": lat, "longitude": lon})

    df = pd.DataFrame(rows).drop_duplicates("station_id")
    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} ETEX stations were parsed from {path}; expected about 168. "
            "The upstream JRC file format may have changed."
        )
    return df.reset_index(drop=True)


def load_etex_concentrations(path: str | Path, station_ids: Iterable[str]) -> pd.DataFrame:
    """Parse JRC ETEX-I ``pmch.dat`` into station x snapshot concentration data.

    Two common historical layouts are handled:
      * one station per line followed by a vector of three-hour concentrations;
      * repeated station records, one concentration per line.

    Missing/negative values are converted to zero because ETEX source-localization uses
    nonnegative measured tracer concentrations.
    """
    path = Path(path)
    known = {_clean_station_token(s): s for s in station_ids}
    records: dict[str, list[list[float]]] = {s: [] for s in known.values()}

    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if not raw.strip() or raw.lstrip().startswith(("#", "!", "%")):
            continue
        tokens = raw.split()
        station = None
        station_pos = None
        for idx, token in enumerate(tokens):
            key = _clean_station_token(token)
            if key in known:
                station = known[key]
                station_pos = idx
                break
        if station is None:
            continue
        # Use numbers after the station token when possible, otherwise all numeric fields.
        tail = " ".join(tokens[station_pos + 1 :]) if station_pos is not None else raw
        vals = _numbers(tail)
        if vals:
            records[station].append(vals)

    usable = {k: v for k, v in records.items() if v}
    if len(usable) < 80:
        raise ValueError(
            f"Only {len(usable)} station records were parsed from {path}; "
            "the ETEX concentration layout was not recognized."
        )

    # Wide layout: one/few line(s) per station and many numbers per line.
    median_records = float(np.median([len(v) for v in usable.values()]))
    median_width = float(np.median([len(v[0]) for v in usable.values()]))
    series: dict[str, np.ndarray] = {}
    if median_records <= 2 and median_width >= 15:
        widths = [len(v[0]) for v in usable.values()]
        common_width = int(pd.Series(widths).mode().iloc[0])
        # ETEX-I has up to 30 three-hour sampling intervals. Historical files may prefix
        # metadata numerics, so retain the right-most 30/common-width values.
        n_time = min(30, common_width)
        for station, recs in usable.items():
            vec = np.asarray(recs[0], float)
            if vec.size >= n_time:
                series[station] = vec[-n_time:]
    else:
        # Long layout: station occurs repeatedly. The final numeric field is concentration.
        counts = [len(v) for v in usable.values()]
        n_time = min(30, int(pd.Series(counts).mode().iloc[0]))
        for station, recs in usable.items():
            if len(recs) >= n_time:
                series[station] = np.asarray([row[-1] for row in recs[:n_time]], float)

    if len(series) < 80:
        raise ValueError("Could not obtain a consistent ETEX station-by-time concentration matrix.")
    n_time = min(len(v) for v in series.values())
    columns = [f"t{3*j:02d}h" for j in range(n_time)]
    df = pd.DataFrame({s: np.asarray(v[:n_time], float) for s, v in series.items()}, index=columns).T
    values = df.to_numpy(float)
    values[~np.isfinite(values)] = 0.0
    values[values < 0.0] = 0.0
    df.iloc[:, :] = values
    return df


def load_etex_i(directory: str | Path) -> EtexData:
    directory = Path(directory)
    stations = load_etex_station_list(directory / "stationlist.950130")
    conc = load_etex_concentrations(directory / "pmch.dat", stations.station_id)
    merged = stations[stations.station_id.isin(conc.index)].copy()
    # Keep the station-list order so graph node indices are deterministic.
    ids = tuple(merged.station_id.tolist())
    matrix = conc.loc[list(ids)].to_numpy(float)
    if matrix.shape[0] < 80 or matrix.shape[1] < 10:
        raise ValueError(f"Unexpected ETEX-I matrix shape: {matrix.shape}")
    return EtexData(
        station_ids=ids,
        latitudes=merged.latitude.to_numpy(float),
        longitudes=merged.longitude.to_numpy(float),
        concentrations=matrix,
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = np.deg2rad([lat1, lat2])
    dp = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def pairwise_haversine(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    n = len(lat)
    d = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = haversine_km(lat[i], lon[i], lat[j], lon[j])
    return d


def build_etex_graph(data: EtexData, k: int = 5, sigma_km: float | None = None) -> nx.Graph:
    """Build a symmetric geographic k-NN graph over real ETEX-I monitoring stations."""
    n = data.n_stations
    if not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < number of stations")
    dist = pairwise_haversine(data.latitudes, data.longitudes)
    neighbor_pairs: set[tuple[int, int]] = set()
    for i in range(n):
        order = np.argsort(dist[i])
        for j in order[1 : k + 1]:
            a, b = sorted((i, int(j)))
            neighbor_pairs.add((a, b))
    edge_distances = np.array([dist[i, j] for i, j in neighbor_pairs], float)
    sigma = float(np.median(edge_distances)) if sigma_km is None else float(sigma_km)
    if sigma <= 0:
        raise ValueError("sigma_km must be positive")
    g = nx.Graph()
    for i, station in enumerate(data.station_ids):
        g.add_node(i, station_id=station, latitude=float(data.latitudes[i]), longitude=float(data.longitudes[i]))
    for i, j in neighbor_pairs:
        dij = dist[i, j]
        g.add_edge(i, j, weight=float(np.exp(-((dij / sigma) ** 2))), distance_km=float(dij))
    if not nx.is_connected(g):
        # Increase connectivity minimally by joining connected components using the closest pair.
        while not nx.is_connected(g):
            comps = [list(c) for c in nx.connected_components(g)]
            best = None
            for a in comps[0]:
                for comp in comps[1:]:
                    for b in comp:
                        candidate = (dist[a, b], a, b)
                        if best is None or candidate < best:
                            best = candidate
            assert best is not None
            dij, i, j = best
            g.add_edge(i, j, weight=float(np.exp(-((dij / sigma) ** 2))), distance_km=float(dij))
    return g


def nearest_station_to_release(data: EtexData) -> int:
    distances = np.array(
        [haversine_km(ETEX_SOURCE_LAT, ETEX_SOURCE_LON, la, lo) for la, lo in zip(data.latitudes, data.longitudes)]
    )
    return int(np.argmin(distances))


def combinatorial_laplacian(g: nx.Graph) -> np.ndarray:
    nodes = list(sorted(g.nodes()))
    a = nx.to_numpy_array(g, nodelist=nodes, weight="weight", dtype=float)
    return np.diag(a.sum(axis=1)) - a


def normalized_laplacian(g: nx.Graph) -> np.ndarray:
    nodes = list(sorted(g.nodes()))
    a = nx.to_numpy_array(g, nodelist=nodes, weight="weight", dtype=float)
    d = a.sum(axis=1)
    invsqrt = np.zeros_like(d)
    mask = d > 0
    invsqrt[mask] = 1.0 / np.sqrt(d[mask])
    return np.eye(len(nodes)) - (invsqrt[:, None] * a) * invsqrt[None, :]


def build_wave_model(g: nx.Graph, max_time: int = 9, safety_factor: float = 1.0) -> WaveModel:
    if max_time < 1:
        raise ValueError("max_time must be >= 1")
    l = combinatorial_laplacian(g)
    eigmax = float(np.linalg.eigvalsh(l)[-1])
    if eigmax <= 0:
        raise ValueError("Graph Laplacian has no positive eigenvalue.")
    nu_dt = float(safety_factor / np.sqrt(eigmax))
    if (nu_dt**2) * eigmax >= 4.0:
        raise ValueError("Unstable recurrence: (nu*dt)^2 lambda_max must be < 4.")
    b = np.eye(l.shape[0]) - 0.5 * (nu_dt**2) * l
    w = [np.eye(l.shape[0]), np.eye(l.shape[0])]
    for _j in range(1, max_time):
        w.append(2.0 * b @ w[-1] - w[-2])
    return WaveModel(g, l, nu_dt, tuple(w[: max_time + 1]))


def stacked_dictionary(model: WaveModel, times: Iterable[int]) -> np.ndarray:
    times = tuple(int(t) for t in times)
    if not times:
        raise ValueError("At least one snapshot time is required.")
    if min(times) < 0 or max(times) >= len(model.propagators):
        raise ValueError("Requested time is outside the precomputed propagator range.")
    return np.vstack([model.propagators[t] for t in times])


def _unit_columns(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    norms = np.linalg.norm(a, axis=0)
    return a / np.maximum(norms, np.finfo(float).eps)


def cosine_dictionary_ranking(dictionary: np.ndarray, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = _unit_columns(dictionary)
    y = np.asarray(observation, float)
    yn = np.linalg.norm(y)
    if yn <= np.finfo(float).eps:
        raise ValueError("Observation has zero norm.")
    scores = np.abs(d.T @ (y / yn))
    order = np.argsort(-scores)
    return order, scores


def modal_alpha(model: WaveModel, times: Iterable[int]) -> float:
    times = tuple(int(t) for t in times)
    evals = np.linalg.eigvalsh(model.laplacian)
    max_t = max(times)
    energies = []
    for lam in evals[1:]:
        phi = [1.0, 1.0]
        for _j in range(1, max_t):
            phi.append(2.0 * (1.0 - 0.5 * model.nu_dt**2 * lam) * phi[-1] - phi[-2])
        energies.append(sum(float(phi[t]) ** 2 for t in times))
    return float(min(energies))


# ----- Peña, Bresson & Vandergheynst (2016): heat kernel + l1 baseline -----

def heat_spectral_components(g: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    l = normalized_laplacian(g)
    return np.linalg.eigh(l)


def heat_kernel_matrix(evals: np.ndarray, evecs: np.ndarray, theta: float) -> np.ndarray:
    return (evecs * np.exp(-float(theta) * evals)) @ evecs.T


def _soft_threshold(z: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


def fista_l1(a: np.ndarray, b: np.ndarray, gamma_over_alpha: float, max_iter: int = 1000, tol: float = 1e-9) -> np.ndarray:
    b = np.asarray(b, float)
    lip = float(np.linalg.norm(a, ord=2) ** 2)
    if lip <= 0:
        raise ValueError("Degenerate diffusion matrix.")
    x = np.zeros(a.shape[1], dtype=float)
    y = x.copy()
    tk = 1.0
    for _ in range(max_iter):
        grad = a.T @ (a @ y - b)
        x_new = _soft_threshold(y - grad / lip, gamma_over_alpha / lip)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        y = x_new + ((tk - 1.0) / t_new) * (x_new - x)
        if np.linalg.norm(x_new - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_new
            break
        x, tk = x_new, t_new
    return x


def pena_l1_heat_source(
    g: nx.Graph,
    observation: np.ndarray,
    gamma: float = 1e-3,
    alpha: float = 1.0,
    theta0: float = 1.0,
    theta_bounds: tuple[float, float] = (0.01, 8.0),
    proximal_eta: float = 1e-2,
    max_outer: int = 20,
    tol: float = 1e-8,
) -> tuple[int, np.ndarray, float, float]:
    """Reproduce the heat-kernel/l1 source-localization objective of Peña et al. (2016)."""
    b = np.asarray(observation, float)
    evals, evecs = heat_spectral_components(g)
    theta = float(theta0)
    prev = np.inf
    x = np.zeros_like(b)
    for _ in range(max_outer):
        a = heat_kernel_matrix(evals, evecs, theta)
        x = fista_l1(a, b, gamma / alpha)
        theta_prev = theta

        def obj(th: float) -> float:
            ah = heat_kernel_matrix(evals, evecs, th)
            residual = ah @ x - b
            return float(
                gamma * np.sum(np.abs(x))
                + 0.5 * alpha * (residual @ residual)
                + 0.5 * proximal_eta * (th - theta_prev) ** 2
            )

        result = minimize_scalar(obj, bounds=theta_bounds, method="bounded", options={"xatol": 1e-7})
        theta = float(result.x)
        value = float(result.fun)
        if abs(prev - value) < tol:
            prev = value
            break
        prev = value
    return int(np.argmax(np.abs(x))), x, theta, float(prev)


def _snapshot_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = np.maximum(x, 0.0)
    n = np.linalg.norm(x)
    return x / n if n > np.finfo(float).eps else x


def geographic_error_km(data: EtexData, estimate: int) -> float:
    return haversine_km(
        ETEX_SOURCE_LAT,
        ETEX_SOURCE_LON,
        float(data.latitudes[estimate]),
        float(data.longitudes[estimate]),
    )


def evaluate_etex_i(
    data: EtexData,
    k: int = 5,
    wave_safety_factor: float = 1.0,
    window_radius: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate graph source recovery on real ETEX-I tracer measurements.

    ETEX observations are physical atmospheric-diffusion measurements, not data generated
    by the manuscript wave recurrence. Therefore cosine-normalized matching is used for
    the wave dictionary and the theorem's bounded-noise guarantee is *not* claimed for
    this real-data model-mismatch experiment.
    """
    g = build_etex_graph(data, k=k)
    mass = np.sum(data.concentrations, axis=0)
    nonzero = np.flatnonzero(mass > 0)
    if nonzero.size == 0:
        raise ValueError("ETEX concentration matrix contains no positive measurements.")
    peak_t = int(np.argmax(mass))
    lo = max(int(nonzero.min()), peak_t - window_radius)
    hi = min(int(nonzero.max()), peak_t + window_radius)
    multi_times_data = tuple(range(lo, hi + 1))
    if len(multi_times_data) < 2:
        multi_times_data = tuple(nonzero[: min(3, len(nonzero))].tolist())

    # Propagator indices are relative snapshots in the selected real-data window.
    single_wave_t = max(1, len(multi_times_data))
    multi_wave_times = tuple(range(1, len(multi_times_data) + 1))
    model = build_wave_model(g, max_time=max(multi_wave_times), safety_factor=wave_safety_factor)

    y_single = _snapshot_normalize(data.concentrations[:, peak_t])
    d_single = model.propagators[single_wave_t]
    order_single, score_single = cosine_dictionary_ranking(d_single, y_single)
    est_single = int(order_single[0])

    y_multi = np.concatenate([_snapshot_normalize(data.concentrations[:, t]) for t in multi_times_data])
    d_multi = stacked_dictionary(model, multi_wave_times)
    order_multi, score_multi = cosine_dictionary_ranking(d_multi, y_multi)
    est_multi = int(order_multi[0])

    # Published-model baseline on the same real peak-concentration snapshot.
    est_pena, x_pena, theta_hat, objective = pena_l1_heat_source(g, y_single)
    order_pena = np.argsort(-np.abs(x_pena))

    # Elementary real-data comparator.
    est_peak = int(np.argmax(data.concentrations[:, peak_t]))

    proxy = nearest_station_to_release(data)
    methods = [
        ("peak_concentration", est_peak, None, None),
        ("single_time_wave", est_single, order_single, score_single),
        ("multi_time_wave", est_multi, order_multi, score_multi),
        ("pena_2016_heat_l1", est_pena, order_pena, np.abs(x_pena)),
    ]
    rows = []
    for method, estimate, order, _scores in methods:
        rank = None
        if order is not None:
            where = np.flatnonzero(np.asarray(order) == proxy)
            rank = int(where[0] + 1) if where.size else None
        rows.append(
            {
                "method": method,
                "estimated_station": data.station_ids[estimate],
                "estimated_node": estimate,
                "geographic_error_km_to_release": geographic_error_km(data, estimate),
                "hop_error_to_nearest_release_station": int(nx.shortest_path_length(g, proxy, estimate)),
                "nearest_release_station_rank": rank,
            }
        )

    metadata: dict[str, object] = {
        "dataset": "ETEX-I European Tracer Experiment",
        "n_stations_used": data.n_stations,
        "n_three_hour_snapshots": data.n_times,
        "graph_edges": g.number_of_edges(),
        "knn_k": k,
        "source_latitude": ETEX_SOURCE_LAT,
        "source_longitude": ETEX_SOURCE_LON,
        "nearest_release_station": data.station_ids[proxy],
        "nearest_release_station_distance_km": geographic_error_km(data, proxy),
        "peak_snapshot_index": peak_t,
        "multi_snapshot_indices": list(multi_times_data),
        "wave_propagator_times": list(multi_wave_times),
        "nu_dt": model.nu_dt,
        "stability_value": float(model.nu_dt**2 * np.linalg.eigvalsh(model.laplacian)[-1]),
        "alpha_multi": modal_alpha(model, multi_wave_times),
        "pena_theta_hat": theta_hat,
        "pena_objective": objective,
        "important_limitation": (
            "ETEX is real atmospheric-diffusion data, not wave-recurrence-generated data; "
            "the manuscript's deterministic wave-noise theorem is therefore not asserted for this experiment."
        ),
    }
    return pd.DataFrame(rows), metadata
