#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nonlocal_apps.downloads import default_cache_dir, ensure_boarding_school
from nonlocal_apps.epidemic import (
    MEMORY_DT_DAYS,
    MEMORY_MU,
    MEMORY_OMEGA,
    PAPER_R0,
    even_cosine_multiplier_closed_form,
    hartley_causal_kernel,
    hartley_multiplier_closed_form,
    hartley_spectral_verification,
    load_boarding_school_csv,
    result_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="1978 boarding-school influenza real-data benchmark.")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "epidemic")
    args = parser.parse_args()

    data_path = ensure_boarding_school(default_cache_dir(ROOT), refresh=args.refresh_data)
    df = load_boarding_school_csv(data_path)
    table, fits = result_table(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "boarding_school_model_comparison.csv", index=False)
    pred = pd.DataFrame({"date": df.date, "observed_in_bed": df.in_bed})
    pred["paper_sir"] = fits["paper_sir"].predicted_infected
    pred["refit_sir"] = fits["refit_sir"].predicted_infected
    pred["hartley_memory"] = fits["hartley_memory"].predicted_infected
    pred.to_csv(args.output_dir / "boarding_school_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(df))
    ax.scatter(x, df.in_bed, label="Observed: boys in bed", zorder=4)
    ax.plot(x, pred.paper_sir, label="Published-parameter classical SIR")
    ax.plot(x, pred.refit_sir, linestyle="--", label="Classical SIR: 2-parameter refit")
    ax.plot(x, pred.hartley_memory, linestyle="-.", label="Mass-normalized Hartley memory")
    ax.set_xlabel("Observation day")
    ax.set_ylabel("Infected / in-bed count")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_app_boarding_fit.png", dpi=240)
    plt.close(fig)

    # Reproduce the manuscript's independent two-sided Hartley multiplier check (alpha=1).
    spectral = hartley_spectral_verification()
    (args.output_dir / "hartley_spectral_verification.json").write_text(json.dumps(spectral, indent=2), encoding="utf-8")
    y = np.linspace(-np.pi / MEMORY_DT_DAYS, np.pi / MEMORY_DT_DAYS, 3000)
    hk = hartley_multiplier_closed_form(y)
    kf = even_cosine_multiplier_closed_form(y)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(y, hk, label=r"$H K_h$ (Hartley kernel)")
    ax.plot(y, kf, label=r"$H K_F$ (cosine counterpart)")
    ax.set_xlabel("Frequency y")
    ax.set_ylabel("Interaction spectrum")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_main_memory_spectrum.png", dpi=240)
    plt.close(fig)

    kernel, kappa_raw = hartley_causal_kernel(
        dt=MEMORY_DT_DAYS, n_terms=5000, mu=MEMORY_MU, omega=MEMORY_OMEGA
    )
    metadata = {
        "population": 763,
        "n_daily_observations": len(df),
        "published_sir_R0": PAPER_R0,
        "hartley_fit_time_step_days": MEMORY_DT_DAYS,
        "hartley_mu_fixed": MEMORY_MU,
        "hartley_omega_fixed": MEMORY_OMEGA,
        "hartley_fit_free_parameters": ["beta", "gamma"],
        "hartley_kernel_normalization": "unit one-sided causal mass",
        "hartley_kernel_alpha_for_fit": "absorbed by unit-mass normalization; not fitted independently",
        "hartley_kernel_raw_one_sided_mass_for_alpha_1": kappa_raw,
        "hartley_kernel_normalized_numeric_mass": float(MEMORY_DT_DAYS * kernel.sum()),
        "spectral_check_uses_alpha": 1.0,
        "data_path": str(data_path),
    }
    (args.output_dir / "epidemic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print("\nHartley multiplier verification:")
    print(json.dumps(spectral, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
