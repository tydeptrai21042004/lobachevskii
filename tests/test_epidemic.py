import numpy as np
import pandas as pd

from nonlocal_apps.epidemic import (
    MEMORY_DT_DAYS,
    PAPER_BETA_PER_DAY,
    PAPER_GAMMA_PER_DAY,
    hartley_causal_kernel,
    hartley_multiplier_closed_form,
    hartley_multiplier_direct_sum,
    hartley_spectral_verification,
    paper_sir_baseline,
    result_table,
    simulate_hartley_memory,
)


def test_hartley_kernel_is_nonnegative_and_mass_normalized():
    kernel, _ = hartley_causal_kernel(dt=MEMORY_DT_DAYS, n_terms=5000, mu=0.5, omega=2.0)
    assert np.min(kernel) >= -1e-12
    assert abs(MEMORY_DT_DAYS * kernel.sum() - 1.0) < 1e-8


def test_memory_sir_preserves_total_population_fraction():
    states, _ = simulate_hartley_memory(0.5, 0.3, n_days=5, infected0=3.0)
    assert np.min(states) >= -1e-12
    assert np.max(np.abs(states.sum(axis=1) - 1.0)) < 1e-10


def test_memory_sir_rejects_parameters_outside_stated_step_condition():
    try:
        simulate_hartley_memory(10.1, 0.3, n_days=3)
    except ValueError as exc:
        assert "positivity step conditions" in str(exc)
    else:
        raise AssertionError("Expected positivity-step validation to reject beta*dt>1")


def test_hartley_multiplier_closed_form_matches_direct_sum():
    y = np.array([-7.0, -2.0, 0.0, 1.5, 4.0, 9.0])
    direct = hartley_multiplier_direct_sum(y, n_cut=20000)
    closed = hartley_multiplier_closed_form(y)
    assert np.max(np.abs(direct - closed)) < 2e-12


def test_manuscript_hartley_spectral_numbers():
    result = hartley_spectral_verification(grid_size=100001)
    assert result["max_direct_sum_abs_error"] < 2e-12
    assert abs(result["max_frequency_asymmetry"] - 2.7849) < 2e-4
    assert abs(result["max_multiplier"] - 4.1682) < 2e-4


def test_published_sir_baseline_accepts_canonical_shape():
    dates = pd.date_range("1978-01-22", periods=14, freq="D")
    df = pd.DataFrame({"date": dates, "in_bed": np.full(14, 3), "convalescent": np.zeros(14)})
    fit = paper_sir_baseline(df)
    assert fit.beta == PAPER_BETA_PER_DAY
    assert fit.gamma == PAPER_GAMMA_PER_DAY
    assert fit.predicted_infected.shape == (14,)



def test_canonical_boarding_school_fit_values_match_manuscript():
    dates = pd.date_range("1978-01-22", periods=14, freq="D")
    in_bed = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4])
    convalescent = np.array([0, 0, 0, 0, 9, 17, 105, 162, 176, 166, 150, 85, 47, 20])
    df = pd.DataFrame({"date": dates, "in_bed": in_bed, "convalescent": convalescent})
    table, _ = result_table(df)

    published = table.iloc[0]
    refit = table.iloc[1]
    memory = table.iloc[2]
    assert abs(published.beta_per_day - 1.6600) < 5e-8
    assert abs(published.gamma_per_day - 0.45454545) < 5e-8
    assert abs(published.rmse_in_bed - 18.4784593) < 5e-6
    assert abs(published.mae_in_bed - 15.6603525) < 5e-6
    assert abs(refit.beta_per_day - 1.6997994) < 5e-6
    assert abs(refit.gamma_per_day - 0.4468659) < 5e-6
    assert abs(refit.rmse_in_bed - 16.6354984) < 5e-6
    assert abs(refit.mae_in_bed - 14.5517648) < 5e-6
    assert abs(memory.beta_per_day - 4.3860122) < 5e-6
    assert abs(memory.gamma_per_day - 0.4987717) < 5e-6
    assert abs(memory.rmse_in_bed - 28.0771523) < 5e-6
    assert abs(memory.mae_in_bed - 19.0417543) < 5e-6
