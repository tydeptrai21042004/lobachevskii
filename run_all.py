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
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the three real-data manuscript applications.")
    parser.add_argument("--refresh-data", action="store_true", help="Redownload all automatically fetched datasets.")
    parser.add_argument("--lanl-file", type=Path, default=None, help="Optional existing LANL five-channel record.")
    parser.add_argument("--lanl-dataset-url", default=None, help="Optional direct LANL archive URL if official hosting changes.")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests after the three applications.")
    args = parser.parse_args()

    common = ["--refresh-data"] if args.refresh_data else []
    mechanical = [sys.executable, "scripts/run_mechanical_lanl.py", *common]
    if args.lanl_file is not None:
        mechanical += ["--lanl-file", str(args.lanl_file)]
    if args.lanl_dataset_url:
        mechanical += ["--dataset-url", args.lanl_dataset_url]

    run(mechanical)
    run([sys.executable, "scripts/run_graph_etex.py", *common])
    run([sys.executable, "scripts/run_epidemic_boarding_school.py", *common])
    if args.run_tests:
        run([sys.executable, "-m", "pytest", "-q"])
    print("\nCompleted. Runtime downloads are in .cache/datasets/ and results are in outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
