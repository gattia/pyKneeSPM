# pyKneeSPM

Statistical Parametric Mapping (SPM) for 3D knee bone surface meshes. Inspired by neuroimaging SPM — finds clusters of statistically significant change on registered bone surfaces rather than relying on manually defined ROIs.

## Library Structure

```
pyKneeSPM/
├── __init__.py              # Public API exports
├── knee_spm.py              # Main SPM pipeline classes (orchestration layer)
├── test_statistics.py        # Vertex-wise statistical tests (z-test, correlation t-test)
├── permutation_test.py       # Permutation schemes for non-parametric inference
├── monte_carlo.py            # Monte Carlo simulation for cluster-extent thresholds
├── clustering.py             # Cluster extraction from thresholded statistic maps (VTK)
├── vtk_functions.py          # VTK utilities (I/O, ICP, scalar transfer)
├── main.py                   # Example data loader
├── scratch_random_code.py    # Scratch/dev code (not used by the library)
└── data/
    └── change_cart_thick.vtk # Example cartilage thickness change mesh
```

## Design & Pipeline

The pipeline follows the standard SPM approach:

1. **Data compilation** (`compile_data`) — Register participant meshes to a common reference using ICP + pyfocusr spectral registration. Transfer scalars to the reference mesh via distance-weighted kNN interpolation. Can be skipped entirely with `add_change_data_directly()` if surfaces are already in correspondence.

2. **Vertex-wise test statistics** (`compute_test_statistics`) — Compute a test statistic at every vertex. Currently: one-sample z-test (`OneSampleZTest`), Pearson correlation t-test (`CorrelationTTest`), independent-samples t-test (`IndependentSamplesTTest`), and one-way ANOVA F-test (`OneWayANOVA`).

3. **Cluster extraction** (`cluster_test_statistics`) — Threshold the statistic map, extract contiguous clusters using VTK connectivity filters, compute cluster surface areas.

4. **Permutation-based cluster-extent inference** (`compute_mc_thresholds`) — Monte Carlo simulation: generate null distributions by permuting data (sign-flip for one-sample, label-shuffle for correlation, group-label shuffle for two-sample), re-run test + clustering on each permutation, collect max cluster sizes. The 95th percentile of this null distribution gives the minimum significant cluster size (family-wise error correction).

### Class Hierarchy

```
SPM (base)
└── SingleStatisticSPM (shared pipeline: thresholding, clustering, MC)
    ├── SimpleTimeDifference — one-sample z-test on change maps
    ├── SimpleCorrelation — correlation between surface scalar and secondary variable
    └── GroupComparison — shared base for group-label-based tests
        ├── TwoSampleComparison — independent-samples t-test between two groups
        └── ANOVAComparison — one-way ANOVA F-test across 2+ groups (one-sided clustering)
```

`MonteCarloThreshold` hierarchy mirrors this:
```
MonteCarloThreshold (base)
└── MonteCarloSingleTest (single-map MC: null distribution, threshold calc)
    ├── MonteCarloClusterOneSampleTest — sign-flip permutations
    ├── MonteCarloClusterCorrelationTest — label-shuffle permutations
    └── MonteCarloClusterGroupTest — group-label shuffle permutations
        ├── MonteCarloClusterTwoSampleTest — two-sample t-test defaults
        └── MonteCarloClusterANOVA — ANOVA defaults (one-sided clustering)
```

`PermutationGroupLabels` is a standalone class (does not inherit `PermutationTest`) that shuffles group labels for both two-sample and ANOVA tests. Uses `sympy.utilities.iterables.multiset_permutations` for efficient exhaustive enumeration when the number of unique permutations is small.

### Data Input Modes

1. **With registration** (requires `pyfocusr`): Provide mesh file paths via `add_change_filepath()` or `add_pre_post_filepaths()`. The library handles ICP alignment, spectral registration, and scalar transfer. Used by `SimpleTimeDifference` and `SimpleCorrelation`.

2. **Pre-aligned numpy array** (no pyfocusr needed): Pass a numpy array directly via `add_change_data_directly(data)` where `data` has shape `(n_participants, n_vertices)` for one-sample/two-sample tests or `(n_participants, n_vertices, 2)` for correlation tests. You must also set a reference mesh via `add_reference_change_mesh()`. This mode skips `compile_data()` entirely.

3. **Pre-aligned mesh files** (`GroupComparison` and subclasses only): Add per-participant mesh files via `add_prealigned_filepath(filepath, group_label, scalar_name)`. Vertex coordinates are validated against the reference mesh on each add — raises if they don't match. Scalars are extracted automatically. This is the recommended path for group comparisons when meshes are already in correspondence.

## Dependencies

Core: `vtk`, `numpy`, `scipy`, `sympy`
Optional (registration only): `pyfocusr` — lazily imported in `compile_data()`, not needed for pre-aligned data.

## Installation

```bash
git clone <repo_url>
cd pykneespm
pip install .
```

## Tests

Run with `pytest`. Tests cover all library modules including the two-sample t-test and ANOVA pipelines (112 tests total). Tests use synthetic VTK meshes — no real data files needed.

## Known Bugs

None currently tracked.

### Fixed 2026-07-26 — `PermutationCorrelationTTest` permutation degeneracy

**SEVERE — invalidated every correlation / scalar-regression cluster p-value produced before this
date.** `perform_permutations()` shuffled the participant vector **once** and then drew every
subsequent "permutation" with `next(itertools.permutations(...))`, which enumerates
lexicographically by position. Only the trailing ~8 elements ever moved: at n=694 with 10,000
draws, **686 of 694 participant positions were identical in every draw**, and no position took
more than 9 distinct values (at n=222: 214 of 222). The null distribution was therefore built from
roughly **one** effective random pairing, so the 95th-percentile cluster-extent threshold was a
single random draw — it moved ~8x across identical re-runs of the same data — and every cluster
p-value landed on the `1 / n_perms` floor.

Only the correlation branch was affected. `PermutationGroupLabels` (two-sample t-test, ANOVA) and
`PermutationOneSampleZTest` already drew independently, so **ANOVA and two-sample results are not
implicated**.

Fixed to draw independent `np.random.permutation`s, with a genuinely exhaustive branch when
`n! <= n_permutations` and the identity retained as row 0 (so the observed statistic is in the
null and `p >= 1/n_perms`, per Phipson & Smyth 2010).

Full evidence, fix and validation: [`.claude/plans/completed/permutation_degeneracy_fix_COMPLETE.md`](.claude/plans/completed/permutation_degeneracy_fix_COMPLETE.md).
Regression tests: `tests/test_permutation_test.py::TestPermutationCorrelationTTestNonDegenerate`.

Previous bugs fixed as of 2026-03-23.
