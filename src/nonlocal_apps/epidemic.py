from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

POPULATION = 763
PAPER_BETA_PER_DAY = 1.66
PAPER_INFECTIOUS_PERIOD_DAYS = 2.2
PAPER_GAMMA_PER_DAY = 1.0 / PAPER_INFECTIOUS_PERIOD_DAYS
PAPER_R0 = PAPER_BETA_PER_DAY / PAPER_GAMMA_PER_DAY
MEMORY_DT_DAYS = 0.1
MEMORY_MU = 0.5
MEMORY_OMEGA = 2.0


@dataclass(frozen=True)
class FitResult:
    beta: float
    gamma: float
    rmse: float
    mae: float
    predicted_infected: np.ndarray


def load_boarding_school_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "in_bed", "convalescent"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(required)}.")
    if len(df) != 14:
        raise ValueError("The canonical outbreaks version has 14 daily observations.")
    if (df[["in_bed", "convalescent"]].to_numpy() < 0).any():
        raise ValueError("Counts must be nonnegative.")
    return df


def _initial_fractions(observed_infected0: float, population: int = POPULATION) -> np.ndarray:
    i0 = float(observed_infected0) / population
    return np.array([1.0 - i0, i0, 0.0], dtype=float)


def simulate_sir(
    beta: float,
    gamma: float,
    days: np.ndarray,
    infected0: float = 3.0,
    population: int = POPULATION,
) -> np.ndarray:
    """Classical frequency-dependent SIR in population fractions."""
    days = np.asarray(days, float)
    y0 = _initial_fractions(infected0, population)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        s, i, _r = y
        inc = beta * s * i
        return np.array([-inc, inc - gamma * i, gamma * i])

    sol = solve_ivp(
        rhs,
        (float(days[0]), float(days[-1])),
        y0,
        t_eval=days,
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def metrics(predicted_counts: np.ndarray, observed_counts: np.ndarray) -> tuple[float, float]:
    p = np.asarray(predicted_counts, float)
    y = np.asarray(observed_counts, float)
    err = p - y
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def paper_sir_baseline(df: pd.DataFrame, population: int = POPULATION) -> FitResult:
    """Reproduce the published boarding-school SIR parameter pair beta=1.66, 1/gamma=2.2 d."""
    days = np.arange(len(df), dtype=float)
    states = simulate_sir(PAPER_BETA_PER_DAY, PAPER_GAMMA_PER_DAY, days, df.in_bed.iloc[0], population)
    pred = population * states[:, 1]
    rmse, mae = metrics(pred, df.in_bed.to_numpy(float))
    return FitResult(PAPER_BETA_PER_DAY, PAPER_GAMMA_PER_DAY, rmse, mae, pred)


def fit_classical_sir(df: pd.DataFrame, population: int = POPULATION) -> FitResult:
    """Two-parameter least-squares SIR refit, initialized at the literature parameter pair."""
    days = np.arange(len(df), dtype=float)
    obs = df.in_bed.to_numpy(float)

    def residual(log_params: np.ndarray) -> np.ndarray:
        beta, gamma = np.exp(log_params)
        states = simulate_sir(beta, gamma, days, obs[0], population)
        return population * states[:, 1] - obs

    x0 = np.log([PAPER_BETA_PER_DAY, PAPER_GAMMA_PER_DAY])
    fit = least_squares(residual, x0=x0, bounds=(np.log([0.02, 0.02]), np.log([10.0, 5.0])), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    beta, gamma = np.exp(fit.x)
    states = simulate_sir(beta, gamma, days, obs[0], population)
    pred = population * states[:, 1]
    rmse, mae = metrics(pred, obs)
    return FitResult(float(beta), float(gamma), rmse, mae, pred)


def hartley_causal_kernel(
    dt: float,
    n_terms: int,
    mu: float = 0.5,
    omega: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Mass-normalized one-sided Hartley-generated kernel.

    The exact infinite one-sided mass is used for normalization, so dt*sum(K_j) -> 1
    as n_terms increases. This removes multiplicative confounding between the kernel amplitude and beta.
    """
    if dt <= 0 or mu <= 0 or omega <= 0 or n_terms < 1:
        raise ValueError("dt, mu, omega and n_terms must be positive.")
    t = dt * np.arange(n_terms, dtype=float)
    raw = np.exp(-mu * t) * (1.0 + (np.cos(omega * t) + np.sin(omega * t)) / np.sqrt(2.0))
    r = np.exp(-mu * dt)
    theta = omega * dt
    denom = 1.0 - 2.0 * r * np.cos(theta) + r * r
    kappa = dt * (
        1.0 / (1.0 - r)
        + (1.0 / np.sqrt(2.0)) * (1.0 - r * np.cos(theta) + r * np.sin(theta)) / denom
    )
    return raw / kappa, float(kappa)


def simulate_hartley_memory(
    beta: float,
    gamma: float,
    n_days: int,
    infected0: float = 3.0,
    population: int = POPULATION,
    dt: float = MEMORY_DT_DAYS,
    mu: float = MEMORY_MU,
    omega: float = MEMORY_OMEGA,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler-Volterra SIR with a unit-mass causal Hartley-generated kernel."""
    if beta < 0 or gamma <= 0:
        raise ValueError("beta must be nonnegative and gamma must be positive.")
    n_steps = int(round((n_days - 1) / dt)) + 1
    if abs((n_steps - 1) * dt - (n_days - 1)) > 1e-10:
        raise ValueError("dt must divide the daily observation interval exactly.")
    # The causal kernel is normalized to unit one-sided mass.  These are exactly the
    # sufficient simplex-preservation step restrictions from the manuscript with kappa_+=1.
    if dt * gamma > 1.0 + 1e-12 or dt * beta > 1.0 + 1e-12:
        raise ValueError("Parameters violate the manuscript positivity step conditions: dt*gamma<=1 and dt*beta<=1.")
    kernel, _ = hartley_causal_kernel(dt, n_steps, mu=mu, omega=omega)
    s = np.empty(n_steps, dtype=float)
    i = np.empty(n_steps, dtype=float)
    rcomp = np.empty(n_steps, dtype=float)
    s[0], i[0], rcomp[0] = _initial_fractions(infected0, population)
    for n in range(n_steps - 1):
        # lambda_n = dt * sum_{m=0}^n K((n-m)dt) I_m
        lam = dt * float(np.dot(kernel[: n + 1], i[n::-1]))
        new_inf = dt * beta * s[n] * lam
        new_rec = dt * gamma * i[n]
        s[n + 1] = s[n] - new_inf
        i[n + 1] = i[n] + new_inf - new_rec
        rcomp[n + 1] = rcomp[n] + new_rec
    return np.column_stack([s, i, rcomp]), kernel


def fit_hartley_memory(
    df: pd.DataFrame,
    population: int = POPULATION,
    dt: float = MEMORY_DT_DAYS,
    mu: float = MEMORY_MU,
    omega: float = MEMORY_OMEGA,
) -> FitResult:
    """Fit beta and gamma only; mu and omega are fixed to the manuscript illustration values."""
    obs = df.in_bed.to_numpy(float)
    daily_index = (np.arange(len(df)) / dt).round().astype(int)

    def residual(log_params: np.ndarray) -> np.ndarray:
        beta, gamma = np.exp(log_params)
        states, _ = simulate_hartley_memory(beta, gamma, len(df), obs[0], population, dt, mu, omega)
        pred = population * states[daily_index, 1]
        return pred - obs

    x0 = np.log([PAPER_BETA_PER_DAY, PAPER_GAMMA_PER_DAY])
    fit = least_squares(
        residual,
        x0=x0,
        bounds=(np.log([0.02, 0.02]), np.log([10.0, 5.0])),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=1000,
    )
    beta, gamma = np.exp(fit.x)
    states, _ = simulate_hartley_memory(beta, gamma, len(df), obs[0], population, dt, mu, omega)
    pred = population * states[daily_index, 1]
    rmse, mae = metrics(pred, obs)
    return FitResult(float(beta), float(gamma), rmse, mae, pred)



def poisson_kernel_chi(xi: np.ndarray | float, *, h: float, chi: float) -> np.ndarray:
    xi = np.asarray(xi, float)
    return (1.0 - chi**2) / (1.0 - 2.0 * chi * np.cos(h * xi) + chi**2)


def hartley_multiplier_closed_form(
    y: np.ndarray | float,
    *,
    h: float = MEMORY_DT_DAYS,
    mu: float = MEMORY_MU,
    omega: float = MEMORY_OMEGA,
    alpha: float = 1.0,
) -> np.ndarray:
    """Closed form of Eq. (3.58) for the two-sided Hartley kernel."""
    y = np.asarray(y, float)
    chi = float(np.exp(-mu * h))
    return alpha * h * (
        poisson_kernel_chi(y, h=h, chi=chi)
        + poisson_kernel_chi(omega - y, h=h, chi=chi) / np.sqrt(2.0)
    )


def even_cosine_multiplier_closed_form(
    y: np.ndarray | float,
    *,
    h: float = MEMORY_DT_DAYS,
    mu: float = MEMORY_MU,
    omega: float = MEMORY_OMEGA,
    alpha: float = 1.0,
) -> np.ndarray:
    """Closed form of the even cosine comparison multiplier, Eq. (3.62)."""
    y = np.asarray(y, float)
    chi = float(np.exp(-mu * h))
    return alpha * h * (
        poisson_kernel_chi(y, h=h, chi=chi)
        + 0.5 * poisson_kernel_chi(omega - y, h=h, chi=chi)
        + 0.5 * poisson_kernel_chi(omega + y, h=h, chi=chi)
    )


def hartley_multiplier_direct_sum(
    y: np.ndarray | float,
    *,
    h: float = MEMORY_DT_DAYS,
    mu: float = MEMORY_MU,
    omega: float = MEMORY_OMEGA,
    alpha: float = 1.0,
    n_cut: int = 20000,
) -> np.ndarray:
    """Direct finite Hartley sum used to numerically verify Eq. (3.58)."""
    y_arr = np.atleast_1d(np.asarray(y, float))
    n = np.arange(-int(n_cut), int(n_cut) + 1, dtype=float)
    t = n * h
    cas_omega = np.cos(omega * t) + np.sin(omega * t)
    kernel = alpha * np.exp(-mu * np.abs(t)) * (1.0 + cas_omega / np.sqrt(2.0))
    out = np.empty_like(y_arr)
    for j, yy in enumerate(y_arr):
        cas_y = np.cos(t * yy) + np.sin(t * yy)
        out[j] = h * float(np.dot(kernel, cas_y))
    return out if np.ndim(y) else out[0]


def hartley_spectral_verification(
    *,
    h: float = MEMORY_DT_DAYS,
    mu: float = MEMORY_MU,
    omega: float = MEMORY_OMEGA,
    alpha: float = 1.0,
    n_cut: int = 20000,
    grid_size: int = 200001,
) -> dict[str, object]:
    """Reproduce the numerical multiplier checks reported in Section 4.3."""
    test_y = np.linspace(-0.75 * np.pi / h, 0.75 * np.pi / h, 6)
    direct = hartley_multiplier_direct_sum(test_y, h=h, mu=mu, omega=omega, alpha=alpha, n_cut=n_cut)
    closed = hartley_multiplier_closed_form(test_y, h=h, mu=mu, omega=omega, alpha=alpha)
    grid = np.linspace(-np.pi / h, np.pi / h, int(grid_size))
    vals = hartley_multiplier_closed_form(grid, h=h, mu=mu, omega=omega, alpha=alpha)
    vals_neg = hartley_multiplier_closed_form(-grid, h=h, mu=mu, omega=omega, alpha=alpha)
    return {
        "test_frequencies": test_y.tolist(),
        "max_direct_sum_abs_error": float(np.max(np.abs(direct - closed))),
        "max_frequency_asymmetry": float(np.max(np.abs(vals - vals_neg))),
        "max_multiplier": float(np.max(vals)),
    }

def result_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, FitResult]]:
    fixed = paper_sir_baseline(df)
    refit = fit_classical_sir(df)
    memory = fit_hartley_memory(df)
    rows = [
        {
            "method": "Keeling-Rohani SIR (published parameters)",
            "beta_per_day": fixed.beta,
            "gamma_per_day": fixed.gamma,
            "R0_or_beta_over_gamma": fixed.beta / fixed.gamma,
            "rmse_in_bed": fixed.rmse,
            "mae_in_bed": fixed.mae,
            "fitted_parameter_count": 0,
        },
        {
            "method": "Classical SIR (2-parameter refit)",
            "beta_per_day": refit.beta,
            "gamma_per_day": refit.gamma,
            "R0_or_beta_over_gamma": refit.beta / refit.gamma,
            "rmse_in_bed": refit.rmse,
            "mae_in_bed": refit.mae,
            "fitted_parameter_count": 2,
        },
        {
            "method": "Hartley causal memory (2-parameter fit; mu=0.5, omega=2 fixed)",
            "beta_per_day": memory.beta,
            "gamma_per_day": memory.gamma,
            "R0_or_beta_over_gamma": memory.beta / memory.gamma,
            "rmse_in_bed": memory.rmse,
            "mae_in_bed": memory.mae,
            "fitted_parameter_count": 2,
        },
    ]
    return pd.DataFrame(rows), {"paper_sir": fixed, "refit_sir": refit, "hartley_memory": memory}
