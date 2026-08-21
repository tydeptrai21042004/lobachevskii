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

from nonlocal_apps.downloads import default_cache_dir, ensure_lanl_three_story
from nonlocal_apps.mechanical import (
    PAPER_NATURAL_FREQUENCIES_HZ,
    PAPER_TC_EQ6_FREQUENCIES_HZ,
    analyze_record,
    generate_synthetic_lanl_fixture,
    load_lanl_record,
)

SUPPORTED = {".mat", ".csv", ".txt", ".dat", ".npy", ".npz"}


def find_usable_record(root: Path, fs: float) -> tuple[Path, object]:
    candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    errors = []
    for path in candidates:
        try:
            return path, load_lanl_record(path, fs=fs)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    preview = "\n".join(errors[:12])
    raise RuntimeError("Downloaded LANL content did not contain a recognized five-channel record.\n" + preview)


def main() -> int:
    parser = argparse.ArgumentParser(description="LANL three-story-frame real-data benchmark.")
    parser.add_argument("--lanl-file", type=Path, default=None, help="Optional already-downloaded five-channel record.")
    parser.add_argument("--dataset-url", default=None, help="Optional direct LANL/mirror archive URL if official hosting changes.")
    parser.add_argument("--refresh-data", action="store_true", help="Redownload cached data.")
    parser.add_argument("--fs", type=float, default=320.0)
    parser.add_argument("--demo-fixture", action="store_true", help="Synthetic smoke-test only; never used by default.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mechanical")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo_fixture:
        record = generate_synthetic_lanl_fixture(fs=args.fs)
        record_path = None
        source_kind = "synthetic_fixture"
    else:
        if args.lanl_file is not None:
            record_path = args.lanl_file
            record = load_lanl_record(record_path, fs=args.fs)
        else:
            downloaded = ensure_lanl_three_story(
                default_cache_dir(ROOT), refresh=args.refresh_data, dataset_url=args.dataset_url
            )
            record_path, record = find_usable_record(downloaded, args.fs)
        source_kind = "real_lanl"

    result = analyze_record(record)
    rows = []
    for idx, ref in enumerate(PAPER_NATURAL_FREQUENCIES_HZ, start=1):
        rows.append(
            {
                "mode": idx,
                "paper_experiment_hz": ref,
                "paper_transmissibility_reference_hz": PAPER_TC_EQ6_FREQUENCIES_HZ[idx - 1],
                "h1_selected_hz": result["h1_selected_hz"][idx - 1],
                "transmissibility_selected_hz": result["tc_selected_hz"][idx - 1],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "mechanical_frequency_summary.csv", index=False)

    f = result["frequency_hz"]
    ft = result["tc_frequency_hz"]
    h1_mag = np.abs(result["h1"])
    tc_mag = np.abs(result["tc_indicator"])
    mask = (f >= 20) & (f <= 80)
    maskt = (ft >= 20) & (ft <= 80)
    h1_norm = h1_mag / max(np.nanmax(h1_mag[mask]), np.finfo(float).eps)
    tc_norm = tc_mag / max(np.nanmax(tc_mag[maskt]), np.finfo(float).eps)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(f[mask], h1_norm[mask], label="Measured force -> floor-3 H1 FRF")
    ax.plot(ft[maskt], tc_norm[maskt], label="Transmissibility-coherence baseline")
    for ref in PAPER_NATURAL_FREQUENCIES_HZ:
        ax.axvline(ref, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized magnitude")
    ax.set_title("LANL three-story frame" if source_kind == "real_lanl" else "Synthetic smoke-test fixture")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "mechanical_lanl_baseline_comparison.png", dpi=220)
    plt.close(fig)

    metadata = {
        "source_kind": source_kind,
        "record_path": None if record_path is None else str(record_path),
        "fs_hz": record.fs,
        "n_samples": record.n_samples,
        "note": "No packaged data are used; real data are downloaded/cached at runtime.",
    }
    (args.output_dir / "mechanical_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
