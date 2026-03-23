# Plan: Test Suite

## Goal

Establish tests for existing functionality before adding new features (t-test, ANOVA). Tests should verify the math is correct and catch regressions.

## Test framework

Use `pytest`. Tests go in a top-level `tests/` directory. Add `pytest` to dev dependencies.

```
tests/
├── conftest.py               # Shared fixtures (synthetic meshes, fake data)
├── test_test_statistics.py    # Vertex-wise stat tests
├── test_permutation_test.py   # Permutation generation
├── test_clustering.py         # Cluster extraction & area
├── test_monte_carlo.py        # End-to-end MC simulation
├── test_vtk_functions.py      # VTK utilities
└── test_knee_spm.py           # Integration: full pipeline with pre-aligned data
```

## Fixtures (`conftest.py`)

### Synthetic VTK mesh

Create a simple mesh in code (e.g., a flat triangulated grid, ~100-500 points) with known geometry. No file I/O needed — build with VTK directly. This gives us known point counts, known cell areas, and predictable clustering behavior.

### Synthetic data with known answers

Generate numpy arrays where we know what the statistical result should be:
- **Uniform positive shift**: all participants have the same positive value at every vertex → z-score should be large and uniform.
- **Zero-mean noise**: random data centered on zero → z-scores should be small / non-significant.
- **Two regions**: one region with a real effect, one with noise → should produce exactly one significant cluster in the effect region.
- **Perfect correlation**: secondary variable = linear function of primary → r ≈ 1, t should be large.
- **No correlation**: secondary variable is random noise → r ≈ 0.

## Test plan by module

### 1. `test_test_statistics.py` — verify the math

**`OneSampleZTest`**:
- Hand-compute z = mean / (std/sqrt(n)) for a small (e.g., 5 participants × 10 vertices) array. Compare to library output. This is the most critical test — verifies the core statistic is correct.
- Verify `idx_not_to_include` vertices are zeroed out.
- Verify non-finite values are replaced with 0.
- Verify result is stored as VTK scalars named `'z_statistic'` on the mesh.

**`CorrelationTTest`**:
- Create data with known Pearson r (e.g., perfectly correlated: `x2 = 2*x1 + 3`). Verify r ≈ 1 and t is large.
- Create uncorrelated data. Verify r ≈ 0.
- Compare correlation output against `np.corrcoef` or `scipy.stats.pearsonr` for a few vertices.
- Compare t-statistic against `scipy.stats.ttest_ind` or manual `t = r * sqrt((n-2)/(1-r^2))`.

**`replace_non_finite_values`**:
- Pass array with NaN, inf, -inf. Verify all replaced. (This will also surface the bug that `replace_value` is ignored.)

### 2. `test_permutation_test.py` — verify permutation properties

**`PermutationOneSampleZTest`**:
- Small n (e.g., 3 participants) where all 2^3 = 8 unique sign permutations are enumerable. Verify we get exactly 8 unique permutations.
- Verify each permutation matrix is diagonal with entries in {-1, +1}.
- Verify that applying sign permutations to data produces correctly sign-flipped values (spot-check a few).
- Larger n: verify output shape is `(n_permutations, n_participants, n_vertices)`.

**`PermutationCorrelationTTest`**:
- Small n (e.g., 3 participants) where all 3! = 6 permutations are enumerable. Verify we get exactly 6.
- Verify the primary variable (column 0) is unchanged across permutations — only the secondary variable (column 1) is shuffled.
- Verify each permutation is a valid reordering of participants (no duplicates, no missing).

### 3. `test_clustering.py` — verify cluster extraction

- Create a mesh with scalars that have two spatially separated high-value regions and low values elsewhere. Threshold and verify two clusters are found.
- Create a mesh with all values below threshold. Verify zero clusters.
- Create a mesh with all values above threshold. Verify one cluster whose area equals the full mesh area.
- Verify `get_areas()` returns positive values that are geometrically plausible (e.g., for a known flat grid, area = n_cells × cell_area).

### 4. `test_monte_carlo.py` — verify MC simulation

These are slower integration-style tests. Use small iterations (e.g., 50-100) and small meshes.

**`MonteCarloClusterOneSampleTest`**:
- Run on pure noise data. Verify threshold cluster size is > 0 (the distribution was built).
- Run on data with a massive effect (all values = 10). Verify the actual cluster size exceeds the threshold (i.e., effect is significant).
- Verify `get_distribution_of_max_clustersizes()` returns a sorted list of length `n_iterations`.
- Verify `get_threshold_clustersize(0.05)` returns the 95th percentile value.

**`MonteCarloClusterCorrelationTest`**:
- Same pattern: noise data should not be significant, strong correlation should be significant.

### 5. `test_vtk_functions.py` — verify utilities

- `read_vtk`: read the bundled example file (`data/change_cart_thick.vtk`), verify it returns a vtkPolyData with > 0 points.
- `clean_polydata`: pass a mesh, verify output has same or fewer points and is valid.
- `get_icp_transform` + `apply_transform`: transform a mesh, verify it moved (points changed). Transform identity case — apply mesh to itself, verify points are close to original.
- `transfer_mesh_scalars_get_weighted_average_n_closest`: create two identical meshes with known scalars, verify transferred scalars match originals.

### 6. `test_knee_spm.py` — integration test with pre-aligned data

Test the full pipeline using `add_change_data_directly()` (no pyfocusr needed):

**`SimpleTimeDifference`**:
- Create synthetic data with a strong localized effect on a synthetic mesh.
- Run `update()`. Verify test statistic map exists, clusters were found, MC thresholds were computed.
- Use small `n_monte_carlo_iterations` (e.g., 100) for speed.

**`SimpleCorrelation`**:
- Same pattern with correlated data.

## Running tests

```bash
pytest tests/ -v
```

For fast iteration during development, skip the slow MC tests:
```bash
pytest tests/ -v -m "not slow"
```

Mark MC / integration tests with `@pytest.mark.slow`.

## Priority order

1. `test_test_statistics.py` — most critical, validates the core math
2. `test_permutation_test.py` — validates the inference foundation
3. `test_clustering.py` — validates cluster extraction
4. `test_vtk_functions.py` — validates utilities
5. `test_monte_carlo.py` — integration of stats + permutations + clustering
6. `test_knee_spm.py` — full pipeline integration

---

## Completion Summary (2026-03-23)

**Result**: 53 tests, all passing. Full suite runs in ~24s (Python 3.11, conda env `pykneespm`).

### What was implemented

All 6 planned test modules were implemented with all planned test cases covered. Final test counts:

| Module | Tests |
|--------|-------|
| `test_test_statistics.py` | 14 |
| `test_permutation_test.py` | 10 |
| `test_clustering.py` | 6 |
| `test_monte_carlo.py` | 6 |
| `test_vtk_functions.py` | 6 |
| `test_knee_spm.py` | 11 |

### Deviations from the plan

**Integration tests went further than planned.** The plan only called for checking that maps/thresholds *exist*. The final `test_knee_spm.py` also:
- Injects a known spatial effect onto a specific quadrant of the mesh
- Verifies the pipeline detects it as significant (cluster area > MC threshold)
- Verifies the significant cluster spatially overlaps the injected effect region (coordinate matching with `_cluster_overlaps_region` helper)
- Verifies pure noise does *not* produce significant clusters (multi-seed false-positive check, 5 seeds, allows at most 1 FP)

**Two mesh fixtures instead of one.** `conftest.py` provides both `flat_grid_mesh` (10x10, 100 pts) for integration tests needing spatial separation, and `small_mesh` (5x5, 25 pts) for faster unit-style tests.

**Negative/two-sided clustering tests** added beyond plan scope.

### Bug fixes required during implementation

Three known bugs had to be fixed for the tests to pass:

1. **`np.int` removed in NumPy 1.24+** — `permutation_test.py:91`, `monte_carlo.py:63,76`. Changed to `int`.
2. **Division by zero in `transfer_mesh_scalars`** — `vtk_functions.py:68`. Added zero-distance guard (`1e16` weight when distance is 0).
3. **`n_participants` attribute missing** — `knee_spm.py:466` referenced `self.n_participants` which doesn't exist when using `add_change_data_directly()`. Changed to `self.change_values.shape[0]`.

Config fixes also required: `pyproject.toml` build-backend and `pymskt` → `mskt` dependency name.

### Notes for future work

- **Cluster coordinate matching**: VTK's `vtkBandedPolyDataContourFilter` interpolates new points along contour boundaries. Cluster meshes do NOT share exact vertex coordinates with the reference mesh. Any test checking cluster spatial location needs tolerance >= 1 grid spacing unit. The current `_cluster_overlaps_region` helper uses `tol=1.1`.
- **`replace_value` bug confirmed**: `test_replace_value_ignored_bug` documents that `replace_non_finite_values()` ignores its `replace_value` parameter (always replaces with 0). Test is written to detect when/if this is fixed.
- **Remaining known bugs not yet fixed**: `GetAray` typo, ICP `SetMaximumNumberOfLandmarks` called after `Update()`, `threshold_cluster_size` init as `{}` vs `None`, double-init of `threshold_test_statistic` dict. These don't block tests but should be addressed.
- **Test speed**: `@pytest.mark.slow` tests (MC/integration) take ~20s. Use `pytest -m "not slow"` for fast iteration (~3s).
