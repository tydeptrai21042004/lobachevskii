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
from nonlocal_apps.epidemic import PAPER_R0, hartley_causal_kernel, load_boarding_school_csv, result_table


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
    ax.set_title("1978 English boarding-school influenza")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "boarding_school_fit_comparison.png", dpi=220)
    plt.close(fig)

    kernel, kappa_raw = hartley_causal_kernel(dt=0.1, n_terms=2000, mu=0.5, omega=2.0)
    metadata = {
        "population": 763,
        "n_daily_observations": len(df),
        "published_sir_R0": PAPER_R0,
        "hartley_kernel_raw_one_sided_mass": kappa_raw,
        "hartley_kernel_normalized_numeric_mass": float(0.1 * kernel.sum()),
        "data_path": str(data_path),
    }
    (args.output_dir / "epidemic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
