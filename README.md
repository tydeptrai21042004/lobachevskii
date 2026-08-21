

This repository is aligned to the final manuscript **Interaction Kernels and Weighted Convolution Bounds in Discrete Dynamics: Wave Response, Graph Source Recovery, and Causal Memory**.


## Applications

| Application | Dataset | What the code reproduces |
|---|---|---|
| Mechanical | **RWTH Aachen two-storey steel-frame shaking-table data** | NN/NNN analytical numbers, isolation-design numbers, RWTH white-noise Welch/H1 response summary and response figure |
| Graph source recovery | **ETEX-I** | peak heuristic, single-time wave, multi-time wave using only `j>=2`, and Peña et al. heat-kernel + L1 baseline |
| Epidemic memory | **1978 English boarding-school influenza** | published SIR baseline, two-parameter SIR refit, mass-normalized Hartley-memory fit, and Hartley multiplier numerical verification |

## Install

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run everything

```bash
python run_all.py
```

Force fresh downloads:

```bash
python run_all.py --refresh-data
```

Run unit tests after all applications:

```bash
python run_all.py --run-tests
```

The default run never silently substitutes synthetic data when a public dataset download fails.

---

## 1. Mechanical — RWTH Aachen two-storey steel frame

```bash
python scripts/run_mechanical_rwth.py
```

The script downloads the official Zenodo record DOI `10.5281/zenodo.10134011`, verifies the published MD5 checksum for `Data_v1.0.0.zip`, and extracts only:

```text
LP02_Whitenoise_001.csv
```

The expected columns are:

```text
Time, Acc_0, Acc_1, Acc_2, Disp_0, Disp_1, Disp_2
```

with `dt = 0.003 s`. The measured table channels (`Acc_0`, `Disp_0`) are inputs and the first/second-floor channels are responses. Welch/H1 estimation uses `nperseg=4096`, 50% overlap, and the manuscript band `0.2–50 Hz`.

You can use an existing local CSV:

```bash
python scripts/run_mechanical_rwth.py --rwth-file path/to/LP02_Whitenoise_001.csv
```

or a direct archive URL if Zenodo routing changes:

```bash
python scripts/run_mechanical_rwth.py --dataset-url "DIRECT_DATA_V1_ZIP_URL"
```

Generated outputs include:

```text
outputs/mechanical_rwth/
  rwth_summary.json
  rwth_manuscript_comparison.json
  fig_app_rwth_response.png
  lattice_band_edges.csv
  lattice_design_numbers.json
  fig_main_dispersion.png
```

A synthetic RWTH-shaped fixture exists only for unit/CI testing and can be run explicitly with `--demo-fixture`; it is never used by the default real-data run.

---

## 2. Graph source recovery — ETEX-I

```bash
python scripts/run_graph_etex.py
```

The official JRC ETEX-I files are downloaded from the JRC ETEX Release 1 archive. The graph uses a symmetric geographic 5-nearest-neighbour construction with Gaussian distance weights.

### No theory change

The theoretical initialization stays exactly `W0=W1=Id`. For the **real-data comparison only**, the selected data window is mapped to

```text
W2, W3, W4, ...
```

rather than `W1, W2, W3, ...`. This prevents the multi-time empirical matcher from receiving the unpropagated identity dictionary as one of its inputs.

The single-time wave comparator uses the propagated index aligned with the peak ETEX snapshot inside the same selected data window.

ETEX is atmospheric-transport data, not wave-generated data. Therefore wave matching remains cosine-normalized and the deterministic bounded-noise theorem is not claimed for this model-mismatch experiment.

Methods:

1. peak-concentration heuristic;
2. single-time polynomial-wave matching;
3. multi-time polynomial-wave matching with post-propagation states only;
4. Peña–Bresson–Vandergheynst heat-kernel + L1 source-localization baseline.

After this correction, rerun the ETEX script and use the newly generated values in the manuscript application table. The theorem itself does not need to change.

---

## 3. Epidemic memory — 1978 boarding-school influenza

```bash
python scripts/run_epidemic_boarding_school.py
```

The code uses:

```text
population = 763
memory integration step h = 0.1 day
mu = 0.5 fixed
omega = 2 fixed
fitted memory parameters = beta, gamma only
```

For the real-data Hartley-memory fit, the one-sided causal kernel is normalized to unit mass,

```text
h * sum(K_j) = 1,
```

so the kernel amplitude is not independently fitted and is not confounded with `beta`.

The same script also independently reproduces the manuscript's two-sided Hartley spectral check with `alpha=1`, including direct-sum versus closed-form agreement, maximum frequency asymmetry, and maximum multiplier.

Generated outputs include:

```text
outputs/epidemic/
  boarding_school_model_comparison.csv
  boarding_school_predictions.csv
  epidemic_metadata.json
  hartley_spectral_verification.json
  fig_app_boarding_fit.png
  fig_main_memory_spectrum.png
```

---

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

The tests check, without network access:

- the manuscript NN/NNN band-edge values;
- the two reported group velocities;
- the isolation threshold and maximum gains;
- RWTH spectral-processing logic on a synthetic fixture;
- preservation of the theoretical graph initialization `W0=W1=Id`;
- exclusion of `W0,W1` from the ETEX empirical matcher;
- graph stability and the Peña baseline solver;
- causal-kernel nonnegativity and unit-mass normalization;
- simplex preservation and positivity-step validation;
- Hartley direct-sum/closed-form agreement;
- the reported Hartley asymmetry and multiplier maxima.

## Code layout

```text
README.md
CORRECTIONS.md
requirements.txt
run_all.py
scripts/
  run_mechanical_rwth.py
  run_graph_etex.py
  run_epidemic_boarding_school.py
src/nonlocal_apps/
  downloads.py
  mechanical.py
  graph_source.py
  epidemic.py
tests/
  test_mechanical.py
  test_graph.py
  test_epidemic.py
```

## Public sources

- RWTH Aachen steel-frame dataset: Lenzen et al., Zenodo DOI `10.5281/zenodo.10134011`.
- ETEX-I: van Dop et al. (1998), *Atmospheric Environment* 32(24), 4089–4094.
- Graph baseline: Peña, Bresson & Vandergheynst (2016), IEEE IVMSP Workshop, DOI `10.1109/IVMSPW.2016.7528230`.
- Boarding-school influenza: Anonymous (1978), *British Medical Journal* 1:578; classical parameter benchmark from Keeling & Rohani.
