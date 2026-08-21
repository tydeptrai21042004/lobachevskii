from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, signal

RWTH_EXPECTED_COLUMNS = ("Time", "Acc_0", "Acc_1", "Acc_2", "Disp_0", "Disp_1", "Disp_2")
RWTH_EXPECTED_DT = 0.003
RWTH_EXPECTED_SAMPLES = 40131
RWTH_NPERSEG = 4096
RWTH_BAND_HZ = (0.2, 50.0)

# Values printed in the final manuscript. They are comparison targets, not hard-coded outputs.
RWTH_MANUSCRIPT_TARGETS = {
    "dominant_response_frequency_hz": 3.2552,
    "acceleration_transfer_norm": 12.6212,
    "displacement_transfer_norm": 18.1514,
    "transmissibility_coherence": 0.9970,
}


@dataclass(frozen=True)
class RWTHRecord:
    time: np.ndarray
    acc_table: np.ndarray
    acc_floor1: np.ndarray
    acc_floor2: np.ndarray
    disp_table: np.ndarray
    disp_floor1: np.ndarray
    disp_floor2: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.time.size)

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.time)))

    @property
    def fs(self) -> float:
        return 1.0 / self.dt


def load_rwth_white_noise_csv(path: str | Path) -> RWTHRecord:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=None, engine="python")
    normalized = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in RWTH_EXPECTED_COLUMNS if c.lower() not in normalized]
    if missing:
        raise ValueError(f"RWTH CSV is missing columns: {missing}; found {list(df.columns)}")

    arrays = {c: df[normalized[c.lower()]].to_numpy(float) for c in RWTH_EXPECTED_COLUMNS}
    matrix = np.column_stack([arrays[c] for c in RWTH_EXPECTED_COLUMNS])
    if not np.isfinite(matrix).all():
        raise ValueError("RWTH record contains NaN or infinite values.")
    if len(matrix) < RWTH_NPERSEG:
        raise ValueError(f"RWTH record is too short: {len(matrix)} samples.")
    if np.any(np.diff(arrays["Time"]) <= 0):
        raise ValueError("RWTH time column must be strictly increasing.")

    record = RWTHRecord(
        time=arrays["Time"],
        acc_table=arrays["Acc_0"],
        acc_floor1=arrays["Acc_1"],
        acc_floor2=arrays["Acc_2"],
        disp_table=arrays["Disp_0"],
        disp_floor1=arrays["Disp_1"],
        disp_floor2=arrays["Disp_2"],
    )
    if not np.isclose(record.dt, RWTH_EXPECTED_DT, rtol=0.0, atol=5e-7):
        raise ValueError(f"Unexpected RWTH time step {record.dt:.9g} s; expected about {RWTH_EXPECTED_DT} s.")
    return record


def generate_synthetic_rwth_fixture(
    n_samples: int = 40131,
    dt: float = 0.003,
    seed: int = 20260821,
) -> RWTHRecord:
    """Deterministic CI fixture only. Never used by the default real-data run."""
    rng = np.random.default_rng(seed)
    fs = 1.0 / dt
    x = rng.normal(size=n_samples)

    def resonant(sig: np.ndarray, f0: float, q: float) -> np.ndarray:
        b, a = signal.iirpeak(f0, q, fs=fs)
        return signal.lfilter(b, a, sig)

    r1 = resonant(x, 3.2552, 20.0)
    r2 = resonant(x, 8.6, 18.0)
    acc1 = 5.0 * r1 + 1.0 * r2 + 0.05 * x
    acc2 = 10.0 * r1 + 1.7 * r2 + 0.04 * x

    # Separate displacement fixture with the same dominant mode. This is only for CI.
    xd = signal.lfilter(*signal.butter(2, 12.0, fs=fs), x)
    d1 = 7.0 * resonant(xd, 3.2552, 20.0) + 0.05 * xd
    d2 = 14.0 * resonant(xd, 3.2552, 20.0) + 0.05 * xd
    t = dt * np.arange(n_samples)
    return RWTHRecord(t, x, acc1, acc2, xd, d1, d2)


def _spectral_args(n: int, nperseg: int = RWTH_NPERSEG) -> tuple[int, int]:
    nperseg = int(min(max(128, nperseg), n))
    return nperseg, nperseg // 2


def h1_frf(
    excitation: np.ndarray,
    response: np.ndarray,
    fs: float,
    nperseg: int = RWTH_NPERSEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch/H1 estimate H1=S_yx/S_xx and input-output magnitude-squared coherence."""
    x = signal.detrend(np.asarray(excitation, float), type="constant")
    y = signal.detrend(np.asarray(response, float), type="constant")
    if x.shape != y.shape:
        raise ValueError("Excitation and response must have the same shape.")
    nperseg, noverlap = _spectral_args(len(x), nperseg)
    kwargs = dict(fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)
    f, sxx = signal.welch(x, **kwargs)
    _, syx = signal.csd(x, y, **kwargs)
    _, syy = signal.welch(y, **kwargs)
    eps = np.finfo(float).tiny
    h1 = syx / np.maximum(sxx, eps)
    coh = np.abs(syx) ** 2 / np.maximum(sxx * syy, eps)
    return f, h1, np.clip(np.real(coh), 0.0, 1.0)


def transmissibility_coherence(
    floor1: np.ndarray,
    floor2: np.ndarray,
    fs: float,
    nperseg: int = RWTH_NPERSEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-response transmissibility coherence |G12|^2/(G11 G22).

    This is the coherence quantity paired with the floor-to-floor transmissibility
    calculation used in the manuscript's RWTH summary.
    """
    x1 = signal.detrend(np.asarray(floor1, float), type="constant")
    x2 = signal.detrend(np.asarray(floor2, float), type="constant")
    if x1.shape != x2.shape:
        raise ValueError("Floor signals must have the same shape.")
    nperseg, noverlap = _spectral_args(len(x1), nperseg)
    kwargs = dict(fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)
    f, g11 = signal.welch(x1, **kwargs)
    _, g22 = signal.welch(x2, **kwargs)
    _, g12 = signal.csd(x1, x2, **kwargs)
    eps = np.finfo(float).tiny
    coh = np.abs(g12) ** 2 / np.maximum(g11 * g22, eps)
    return f, np.clip(np.real(coh), 0.0, 1.0)


def multi_floor_transfer(
    table_signal: np.ndarray,
    floor1: np.ndarray,
    floor2: np.ndarray,
    fs: float,
    nperseg: int = RWTH_NPERSEG,
) -> dict[str, np.ndarray]:
    f, h1_1, coh1 = h1_frf(table_signal, floor1, fs, nperseg)
    f2, h1_2, coh2 = h1_frf(table_signal, floor2, fs, nperseg)
    if not np.array_equal(f, f2):
        raise RuntimeError("Inconsistent Welch frequency grids.")
    norm = np.sqrt(np.abs(h1_1) ** 2 + np.abs(h1_2) ** 2)
    return {"frequency_hz": f, "floor1_h1": h1_1, "floor2_h1": h1_2, "norm": norm, "coh1": coh1, "coh2": coh2}


def analyze_rwth_record(
    record: RWTHRecord,
    *,
    nperseg: int = RWTH_NPERSEG,
    band_hz: tuple[float, float] = RWTH_BAND_HZ,
) -> dict[str, object]:
    """Compute the manuscript RWTH summary from table input and two floor responses."""
    acc = multi_floor_transfer(record.acc_table, record.acc_floor1, record.acc_floor2, record.fs, nperseg)
    disp = multi_floor_transfer(record.disp_table, record.disp_floor1, record.disp_floor2, record.fs, nperseg)
    f = np.asarray(acc["frequency_hz"])
    fd = np.asarray(disp["frequency_hz"])
    if not np.array_equal(f, fd):
        raise RuntimeError("Acceleration/displacement frequency grids differ.")
    mask = (f >= band_hz[0]) & (f <= band_hz[1])
    if not np.any(mask):
        raise ValueError("Requested frequency band has no Welch bins.")
    band_indices = np.flatnonzero(mask)
    acc_norm = np.asarray(acc["norm"], float)
    disp_norm = np.asarray(disp["norm"], float)
    idx_acc = int(band_indices[np.nanargmax(acc_norm[mask])])
    idx_disp = int(band_indices[np.nanargmax(disp_norm[mask])])

    fc, tc = transmissibility_coherence(record.acc_floor1, record.acc_floor2, record.fs, nperseg)
    if not np.array_equal(f, fc):
        raise RuntimeError("Coherence frequency grid differs from H1 grid.")

    summary = {
        "dominant_response_frequency_hz": float(f[idx_acc]),
        "acceleration_transfer_norm": float(acc_norm[idx_acc]),
        "displacement_transfer_norm": float(disp_norm[idx_disp]),
        "displacement_peak_frequency_hz": float(f[idx_disp]),
        "transmissibility_coherence": float(tc[idx_acc]),
        "floor1_input_coherence_at_dominant": float(np.asarray(acc["coh1"])[idx_acc]),
        "floor2_input_coherence_at_dominant": float(np.asarray(acc["coh2"])[idx_acc]),
    }
    return {"summary": summary, "acceleration": acc, "displacement": disp, "transmissibility_coherence_curve": tc}


# ---------- Analytical NN/NNN numbers printed in Section 4.1 ----------

def lambda_hat_2(theta: np.ndarray | float, k0: float, k1: float, k2: float) -> np.ndarray:
    theta = np.asarray(theta, float)
    return k0 + 4.0 * k1 * np.sin(theta / 2.0) ** 2 + 4.0 * k2 * np.sin(theta) ** 2


def nn_nnn_band_edge(zeta: float, *, k0: float = 0.36, k1: float = 1.0, m: float = 1.0) -> dict[str, float | None]:
    k2 = float(zeta) * k1
    if k2 <= k1 / 4.0:
        theta_star = None
        lambda_max = k0 + 4.0 * k1
    else:
        theta_star = float(np.arccos(-k1 / (4.0 * k2)))
        lambda_max = k0 + 2.0 * k1 + 4.0 * k2 + k1**2 / (4.0 * k2)
    return {
        "zeta": float(zeta),
        "lambda_min": float(k0),
        "lambda_max": float(lambda_max),
        "omega_max": float(np.sqrt(lambda_max / m)),
        "theta_star": theta_star,
    }


def group_velocity(theta: float, zeta: float, *, h: float = 1.0, k0: float = 0.36, k1: float = 1.0, m: float = 1.0) -> float:
    k2 = zeta * k1
    lam = float(lambda_hat_2(theta, k0, k1, k2))
    omega = np.sqrt(lam / m)
    return float(h * np.sin(theta) * (k1 + 4.0 * k2 * np.cos(theta)) / (m * omega))


def exact_gain(omega: np.ndarray | float, *, m: float, c: float, k0: float, lambda_max: float) -> np.ndarray:
    omega = np.asarray(omega, float)
    x = m * omega**2
    d = np.where(x < k0, k0 - x, np.where(x > lambda_max, x - lambda_max, 0.0))
    return 1.0 / np.sqrt(d**2 + (c * omega) ** 2)


def isolation_required_k0(
    omega_a: float,
    omega_b: float,
    *,
    m: float = 1.0,
    c: float = 0.08,
    gain_tol: float = 2.0,
) -> float:
    if not 0 < omega_a <= omega_b:
        raise ValueError("Require 0 < omega_a <= omega_b.")

    def requirement(omega: float) -> float:
        rad = max(gain_tol ** -2 - (c * omega) ** 2, 0.0)
        return m * omega**2 + np.sqrt(rad)

    result = optimize.minimize_scalar(lambda w: -requirement(w), bounds=(omega_a, omega_b), method="bounded")
    candidates = [requirement(omega_a), requirement(omega_b), requirement(float(result.x))]
    return float(max(candidates))


def max_gain_interval(
    omega_a: float,
    omega_b: float,
    *,
    m: float,
    c: float,
    k0: float,
    lambda_max: float,
) -> float:
    result = optimize.minimize_scalar(
        lambda w: -float(exact_gain(w, m=m, c=c, k0=k0, lambda_max=lambda_max)),
        bounds=(omega_a, omega_b),
        method="bounded",
        options={"xatol": 1e-14},
    )
    vals = [
        float(exact_gain(omega_a, m=m, c=c, k0=k0, lambda_max=lambda_max)),
        float(exact_gain(omega_b, m=m, c=c, k0=k0, lambda_max=lambda_max)),
        float(exact_gain(float(result.x), m=m, c=c, k0=k0, lambda_max=lambda_max)),
    ]
    return max(vals)


def manuscript_lattice_numbers() -> dict[str, object]:
    zetas = [0.10, 0.25, 0.40, 0.50, 0.70]
    bands = [nn_nnn_band_edge(z) for z in zetas]
    vg = {"zeta_0.10": group_velocity(2.5, 0.10), "zeta_0.50": group_velocity(2.5, 0.50)}
    k0_req = isolation_required_k0(0.35, 0.50)
    lam036 = nn_nnn_band_edge(0.50, k0=0.36)["lambda_max"]
    lam080 = nn_nnn_band_edge(0.50, k0=0.80)["lambda_max"]
    gains = {
        "k0_0.36": max_gain_interval(0.35, 0.50, m=1.0, c=0.08, k0=0.36, lambda_max=float(lam036)),
        "k0_0.80": max_gain_interval(0.35, 0.50, m=1.0, c=0.08, k0=0.80, lambda_max=float(lam080)),
    }
    return {"band_edges": bands, "group_velocity_theta_2.5": vg, "required_k0": k0_req, "max_gains": gains}
