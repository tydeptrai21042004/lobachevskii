from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import signal

PAPER_NATURAL_FREQUENCIES_HZ = np.array([30.70, 54.20], dtype=float)
PAPER_TC_EQ6_FREQUENCIES_HZ = np.array([30.8230, 54.1016], dtype=float)


@dataclass(frozen=True)
class MechanicalRecord:
    force: np.ndarray
    base_accel: np.ndarray
    floor1_accel: np.ndarray
    floor2_accel: np.ndarray
    floor3_accel: np.ndarray
    fs: float = 320.0

    @property
    def n_samples(self) -> int:
        return int(self.force.size)

    def as_matrix(self) -> np.ndarray:
        return np.column_stack(
            [self.force, self.base_accel, self.floor1_accel, self.floor2_accel, self.floor3_accel]
        )


def _orient_five_channels(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2-D numeric array, got shape {a.shape}.")
    if a.shape[1] == 5:
        out = a
    elif a.shape[0] == 5:
        out = a.T
    elif a.shape[1] > 5:
        # Some mirrors append metadata columns. Use the first five only when samples are rows.
        out = a[:, :5]
    elif a.shape[0] > 5 and a.shape[1] < 5:
        raise ValueError(f"Array has fewer than five channels: {a.shape}.")
    else:
        raise ValueError(f"Could not orient a five-channel record from shape {a.shape}.")
    if out.shape[0] < 512:
        raise ValueError(f"Record is too short for spectral estimation: {out.shape[0]} samples.")
    if not np.isfinite(out).all():
        raise ValueError("Record contains NaN or infinite values.")
    return out


def load_lanl_record(path: str | Path, fs: float = 320.0) -> MechanicalRecord:
    """Load one five-channel LANL nonlinear-frame realization.

    Accepted formats: CSV, TXT/DAT, NPY, NPZ, MAT. The expected channel order is
    force, base acceleration, floor-1, floor-2, floor-3 acceleration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        # First try numeric CSV with or without a header.
        try:
            frame = pd.read_csv(path)
            numeric = frame.select_dtypes(include=[np.number]).to_numpy()
            if numeric.shape[1] < 5:
                raise ValueError
            a = numeric
        except Exception:
            a = np.loadtxt(path, delimiter=",")
    elif ext in {".txt", ".dat"}:
        try:
            a = np.loadtxt(path)
        except ValueError:
            a = np.loadtxt(path, delimiter=",")
    elif ext == ".npy":
        a = np.load(path)
    elif ext == ".npz":
        with np.load(path) as z:
            keys = list(z.keys())
            if not keys:
                raise ValueError(f"No arrays in {path}.")
            a = z[keys[0]]
    elif ext == ".mat":
        raw = spio.loadmat(path)
        candidates = []
        for key, value in raw.items():
            if key.startswith("__") or not isinstance(value, np.ndarray) or value.ndim != 2:
                continue
            if 5 in value.shape or value.shape[1] >= 5:
                candidates.append(value)
        if not candidates:
            raise ValueError("No suitable 2-D five-channel numeric array found in MAT file.")
        # Prefer the array with the largest number of elements.
        a = max(candidates, key=lambda x: x.size)
    else:
        raise ValueError(f"Unsupported LANL record extension: {ext}")

    a = _orient_five_channels(a)
    return MechanicalRecord(*(a[:, i].astype(float) for i in range(5)), fs=float(fs))


def generate_synthetic_lanl_fixture(
    n_samples: int = 8192,
    fs: float = 320.0,
    seed: int = 20260821,
) -> MechanicalRecord:
    """Generate a deterministic *synthetic* five-channel CI fixture.

    It imitates two resonances near the published LANL natural frequencies. It is only
    for unit/smoke tests and must never be reported as measured LANL data.
    """
    rng = np.random.default_rng(seed)
    force = rng.normal(size=n_samples)

    def resonant(x: np.ndarray, f0: float, q: float) -> np.ndarray:
        b, a = signal.iirpeak(f0, q, fs=fs)
        return signal.lfilter(b, a, x)

    r1 = resonant(force, 30.70, 24.0)
    r2 = resonant(force, 54.20, 28.0)
    eps = lambda scale: scale * rng.normal(size=n_samples)
    base = 0.35 * r1 + 0.20 * r2 + 0.02 * force + eps(0.005)
    f1 = 0.65 * r1 + 0.45 * r2 + 0.01 * force + eps(0.005)
    f2 = 1.00 * r1 + 0.80 * r2 + eps(0.005)
    f3 = 1.35 * r1 + 1.25 * r2 + eps(0.005)
    return MechanicalRecord(force, base, f1, f2, f3, fs=fs)


def _spectral_args(n: int, nperseg: int | None) -> tuple[int, int]:
    if nperseg is None:
        nperseg = min(2048, n)
    nperseg = int(min(max(128, nperseg), n))
    noverlap = nperseg // 2
    return nperseg, noverlap


def welch_h1_frf(
    force: np.ndarray,
    response: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H1 FRF estimate H1 = S_yx/S_xx using Welch-averaged spectra."""
    force = signal.detrend(np.asarray(force, float), type="constant")
    response = signal.detrend(np.asarray(response, float), type="constant")
    nperseg, noverlap = _spectral_args(len(force), nperseg)
    f, pxx = signal.welch(force, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    _, pyx = signal.csd(force, response, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    _, pyy = signal.welch(response, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    eps = np.finfo(float).eps
    h1 = pyx / np.maximum(pxx, eps)
    coherence = np.abs(pyx) ** 2 / np.maximum(pxx * pyy, eps)
    return f, h1, np.clip(np.real(coherence), 0.0, 1.0)


def zhou_transmissibility_eq6(
    xi: np.ndarray,
    xj: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the two-point transmissibility indicator in Zhou et al. (2017), Eq. (6).

    The paper defines T1(i,j)=Gij/Gjj and T2(i,j)=Gii/Gji, and the frequency
    extraction function is 1/(T2-T1). This implementation computes those spectral
    estimators directly using Welch-averaged auto/cross spectra.
    """
    xi = signal.detrend(np.asarray(xi, float), type="constant")
    xj = signal.detrend(np.asarray(xj, float), type="constant")
    if xi.shape != xj.shape:
        raise ValueError("xi and xj must have the same length.")
    nperseg, noverlap = _spectral_args(len(xi), nperseg)
    f, gii = signal.welch(xi, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    _, gjj = signal.welch(xj, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    # scipy.csd(xj, xi) = conj(Xj) Xi, matching G_ij under the usual convention.
    _, gij = signal.csd(xj, xi, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    gji = np.conjugate(gij)
    eps = 1e-14 * max(1.0, float(np.nanmax(gii)), float(np.nanmax(gjj)))
    t1 = gij / (gjj + eps)
    t2 = gii / (gji + eps)
    denom = t2 - t1
    indicator = np.zeros_like(denom, dtype=complex)
    mask = np.abs(denom) > eps
    indicator[mask] = 1.0 / denom[mask]
    tc = np.abs(gij) ** 2 / np.maximum(gii * gjj, eps)
    return f, indicator, np.clip(np.real(tc), 0.0, 1.0)


def peak_candidates(
    f: np.ndarray,
    magnitude: np.ndarray,
    band: tuple[float, float] = (20.0, 80.0),
    prominence_fraction: float = 0.03,
    min_distance_hz: float = 1.0,
) -> np.ndarray:
    f = np.asarray(f, float)
    mag = np.asarray(magnitude, float)
    mask = (f >= band[0]) & (f <= band[1]) & np.isfinite(mag)
    fb = f[mask]
    mb = mag[mask]
    if fb.size < 3 or np.nanmax(mb) <= 0:
        return np.array([], dtype=float)
    df = float(np.median(np.diff(fb)))
    distance = max(1, int(round(min_distance_hz / max(df, 1e-12))))
    peaks, _ = signal.find_peaks(
        mb,
        prominence=max(np.nanmax(mb) * prominence_fraction, np.finfo(float).eps),
        distance=distance,
    )
    return fb[peaks]


def nearest_reference_peaks(candidates: Iterable[float], references: Iterable[float]) -> np.ndarray:
    cand = np.asarray(list(candidates), float)
    refs = np.asarray(list(references), float)
    if cand.size == 0:
        return np.full(refs.shape, np.nan)
    chosen = []
    available = list(cand)
    for ref in refs:
        idx = int(np.argmin(np.abs(np.asarray(available) - ref)))
        chosen.append(float(available.pop(idx)))
        if not available:
            available = list(cand)
    return np.asarray(chosen)


def analyze_record(record: MechanicalRecord) -> dict[str, object]:
    """Compute a standard H1 FRF and the Zhou et al. two-point Eq.(6) baseline."""
    f, h1, coh = welch_h1_frf(record.force, record.floor3_accel, record.fs)
    ft, fun, tc = zhou_transmissibility_eq6(record.floor3_accel, record.base_accel, record.fs)
    h1_peaks = peak_candidates(f, np.abs(h1))
    tc_peaks = peak_candidates(ft, np.abs(fun), prominence_fraction=0.01)
    h1_sel = nearest_reference_peaks(h1_peaks, PAPER_NATURAL_FREQUENCIES_HZ)
    tc_sel = nearest_reference_peaks(tc_peaks, PAPER_NATURAL_FREQUENCIES_HZ)
    return {
        "frequency_hz": f,
        "h1": h1,
        "h1_coherence": coh,
        "tc_frequency_hz": ft,
        "tc_indicator": fun,
        "tc_coherence": tc,
        "h1_candidates_hz": h1_peaks,
        "tc_candidates_hz": tc_peaks,
        "h1_selected_hz": h1_sel,
        "tc_selected_hz": tc_sel,
    }
