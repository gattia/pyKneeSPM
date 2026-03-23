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
