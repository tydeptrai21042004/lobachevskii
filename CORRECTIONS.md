# Code corrections made for the final manuscript

This revision changes **implementation/reproducibility only**. It does not change the manuscript's mathematical theory.

1. **Mechanical dataset alignment**
   - Removed the obsolete LANL manuscript runner.
   - Added the RWTH Aachen `LP02_Whitenoise_001.csv` pipeline used by the final manuscript.
   - Added checksum-verified Zenodo download and a local-file override.
   - Added code for the manuscript NN/NNN band-edge, group-velocity, and isolation-design numerical values.

2. **ETEX empirical fairness**
   - Preserved `W0=W1=Id` exactly.
   - Changed only the application mapping so ETEX snapshots use `W2,W3,...`.
   - The single-time comparator is aligned with the peak snapshot inside the same window.

3. **Epidemic reproducibility**
   - Made `h=0.1 day`, `mu=0.5`, `omega=2`, and the unit one-sided causal-mass normalization explicit in outputs.
   - Added positivity-step validation consistent with the existing theorem.
   - Added the independent two-sided Hartley multiplier verification reported in the manuscript.

4. **Tests**
   - Added regression tests for manuscript analytical numbers and Hartley spectral numbers.
   - Added explicit tests confirming the graph theory is unchanged while the ETEX experiment excludes identity propagators.
