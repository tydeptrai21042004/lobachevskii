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

from nonlocal_apps.downloads import default_cache_dir, ensure_rwth_white_noise
from nonlocal_apps.mechanical import (
    RWTH_EXPECTED_SAMPLES,
    RWTH_MANUSCRIPT_TARGETS,
    analyze_rwth_record,
    generate_synthetic_rwth_fixture,
    lambda_hat_2,
    load_rwth_white_noise_csv,
    manuscript_lattice_numbers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="RWTH two-storey steel-frame manuscript benchmark.")
    parser.add_argument("--refresh-data", action="store_true", help="Redownload the Zenodo archive.")
    parser.add_argument("--rwth-file", type=Path, default=None, help="Optional existing LP02_Whitenoise_001.csv.")
    parser.add_argument("--dataset-url", default=None, help="Optional direct Data_v1.0.0.zip URL.")
    parser.add_argument("--demo-fixture", action="store_true", help="Synthetic CI/smoke fixture only; never used by default.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mechanical_rwth")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo_fixture:
        record = generate_synthetic_rwth_fixture()
        record_path = None
        source_kind = "synthetic_fixture"
    else:
        record_path = args.rwth_file or ensure_rwth_white_noise(
            default_cache_dir(ROOT), refresh=args.refresh_data, archive_url=args.dataset_url
        )
        record = load_rwth_white_noise_csv(record_path)
        source_kind = "real_rwth"

    analysis = analyze_rwth_record(record)
    summary = dict(analysis["summary"])
    summary["source_kind"] = source_kind
    summary["n_samples"] = record.n_samples
    summary["dt_s"] = record.dt
    summary["fs_hz"] = record.fs
    summary["record_path"] = None if record_path is None else str(record_path)
    summary["expected_manuscript_samples"] = RWTH_EXPECTED_SAMPLES
    (args.output_dir / "rwth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    comparison = {}
    for key, target in RWTH_MANUSCRIPT_TARGETS.items():
        value = float(summary[key])
        comparison[key] = {"computed": value, "manuscript_target": target, "difference": value - target}
    (args.output_dir / "rwth_manuscript_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    f = np.asarray(analysis["acceleration"]["frequency_hz"], float)
    norm = np.asarray(analysis["acceleration"]["norm"], float)
    mask = (f >= 0.2) & (f <= 50.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(f[mask], norm[mask], label="Measured multi-floor acceleration transfer norm")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Transfer magnitude")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_app_rwth_response.png", dpi=240)
    plt.close(fig)

    lattice = manuscript_lattice_numbers()
    pd.DataFrame(lattice["band_edges"]).to_csv(args.output_dir / "lattice_band_edges.csv", index=False)
    (args.output_dir / "lattice_design_numbers.json").write_text(json.dumps(lattice, indent=2), encoding="utf-8")

    theta = np.linspace(0.0, np.pi, 1200)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for zeta in [0.00, 0.15, 0.25, 0.40, 0.70]:
        omega = np.sqrt(lambda_hat_2(theta, 0.36, 1.0, zeta))
        ax.plot(theta / np.pi, omega, label=rf"$k_2/k_1={zeta:.2f}$")
    ax.set_xlabel(r"Normalized wavenumber $\theta/\pi$")
    ax.set_ylabel(r"Natural frequency $\widehat{\Omega}_2(\theta)$")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_main_dispersion.png", dpi=240)
    plt.close(fig)

    print("RWTH measured summary")
    for key in ["dominant_response_frequency_hz", "acceleration_transfer_norm", "displacement_transfer_norm", "transmissibility_coherence"]:
        print(f"  {key}: {summary[key]:.8g}")
    print("\nAnalytical lattice values")
    print(pd.DataFrame(lattice["band_edges"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
