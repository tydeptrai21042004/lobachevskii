import numpy as np
import pandas as pd

from nonlocal_apps.epidemic import (
    PAPER_BETA_PER_DAY,
    PAPER_GAMMA_PER_DAY,
    hartley_causal_kernel,
    paper_sir_baseline,
    simulate_hartley_memory,
)


def test_hartley_kernel_is_nonnegative_and_mass_normalized():
    dt = 0.1
    kernel, _ = hartley_causal_kernel(dt=dt, n_terms=5000, mu=0.5, omega=2.0)
    assert np.min(kernel) >= -1e-12
    assert abs(dt * kernel.sum() - 1.0) < 1e-8


def test_memory_sir_preserves_total_population_fraction():
    states, _ = simulate_hartley_memory(0.5, 0.3, n_days=5, infected0=3.0)
    assert np.max(np.abs(states.sum(axis=1) - 1.0)) < 1e-10


def test_published_sir_baseline_accepts_canonical_shape():
    dates = pd.date_range("1978-01-22", periods=14, freq="D")
    df = pd.DataFrame({"date": dates, "in_bed": np.full(14, 3), "convalescent": np.zeros(14)})
    fit = paper_sir_baseline(df)
    assert fit.beta == PAPER_BETA_PER_DAY
    assert fit.gamma == PAPER_GAMMA_PER_DAY
    assert fit.predicted_infected.shape == (14,)
