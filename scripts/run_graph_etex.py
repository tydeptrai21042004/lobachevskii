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

from nonlocal_apps.downloads import default_cache_dir, ensure_etex_i
from nonlocal_apps.graph_source import ETEX_SOURCE_LAT, ETEX_SOURCE_LON, evaluate_etex_i, load_etex_i


def main() -> int:
    parser = argparse.ArgumentParser(description="ETEX-I real-data graph source-recovery benchmark.")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--k", type=int, default=5, help="Geographic k-NN graph degree target.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "graph_etex")
    args = parser.parse_args()

    dataset_dir = ensure_etex_i(default_cache_dir(ROOT), refresh=args.refresh_data)
    data = load_etex_i(dataset_dir)
    results, metadata = evaluate_etex_i(data, k=args.k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "etex_source_recovery_results.csv", index=False)
    (args.output_dir / "etex_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    frame = results.sort_values("geographic_error_km_to_release", ascending=False)
    ax.barh(frame.method, frame.geographic_error_km_to_release)
    ax.set_xlabel("Localization error to Monterfil release (km)")
    ax.set_title(f"ETEX-I source localization ({data.n_stations} real stations)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "etex_localization_error.png", dpi=220)
    plt.close(fig)

    print(results.to_string(index=False))
    print(f"\nKnown ETEX-I release: lat={ETEX_SOURCE_LAT:.4f}, lon={ETEX_SOURCE_LON:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
