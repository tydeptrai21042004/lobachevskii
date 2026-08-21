#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the three real-data applications.")
    parser.add_argument("--refresh-data", action="store_true", help="Redownload automatically fetched datasets.")
    parser.add_argument("--rwth-file", type=Path, default=None, help="Optional existing LP02_Whitenoise_001.csv.")
    parser.add_argument("--rwth-dataset-url", default=None, help="Optional direct RWTH Data_v1.0.0.zip URL.")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests after the three applications.")
    args = parser.parse_args()

    common = ["--refresh-data"] if args.refresh_data else []
    mechanical = [sys.executable, "scripts/run_mechanical_rwth.py", *common]
    if args.rwth_file is not None:
        mechanical += ["--rwth-file", str(args.rwth_file)]
    if args.rwth_dataset_url:
        mechanical += ["--dataset-url", args.rwth_dataset_url]

    run(mechanical)
    run([sys.executable, "scripts/run_graph_etex.py", *common])
    run([sys.executable, "scripts/run_epidemic_boarding_school.py", *common])
    if args.run_tests:
        run([sys.executable, "-m", "pytest", "-q"])
    print("\nCompleted. Runtime downloads are in .cache/datasets/ and results are in outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
