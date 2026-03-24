# Plan: Group Comparison Tests (Two-Sample T-Test & One-Way ANOVA)

## Scope

**Phase 1 (implement now):** Two-sample t-test — `IndependentSamplesTTest`, `PermutationGroupLabels`, `MonteCarloClusterGroupTest`, `MonteCarloClusterTwoSampleTest`, `GroupComparison`, `TwoSampleComparison`. Plus tests.

**Phase 2 (implement later):** ANOVA — `OneWayANOVA`, `MonteCarloClusterANOVA`, `ANOVAComparison`. The Phase 1 shared classes (`PermutationGroupLabels`, `MonteCarloClusterGroupTest`, `GroupComparison`) are designed to support this with minimal additional code. See "ANOVA Extension" section for design details — do NOT implement in Phase 1.

**Not in scope:** Registration path (`compile_data` via pyfocusr). Left as inherited placeholder. Pre-aligned data modes only.

## Method

Compare a scalar value (e.g., cartilage thinning rate) between two groups at every vertex on a registered surface mesh. Uses permutation-based cluster-extent inference to control for multiple comparisons.

### Statistical approach

1. At each vertex, compute an independent-samples t-statistic between group A and group B.
2. Threshold the t-statistic map (e.g., |t| > 2.33) and extract contiguous clusters.
3. Measure each cluster's surface area.
4. Determine whether each cluster is large enough to be significant using a null distribution of cluster sizes built from permutations.

### Permutation scheme

For each of N permutations (e.g., 10,000):
- Randomly shuffle group labels across **participants** (each participant's entire mesh is randomly assigned to group A or B, preserving group sizes). The per-vertex data within each participant is never rearranged.
- Recompute the vertex-wise t-statistic map from the shuffled group assignments.
- Threshold and cluster the shuffled map.
- Record the largest cluster size.

This builds a null distribution of max cluster sizes. The 95th percentile (or whatever alpha) gives the minimum cluster area for significance.

### Inputs — three data input modes

**Mode A: Pre-aligned meshes (recommended initial path)** — `add_prealigned_filepath(filepath, group_label, scalar_name=...)`
- Add one mesh file per participant. Each mesh must have the same vertex positions (in correspondence).
- Vertex coordinates are validated against the reference mesh on each add — raises if they don't match.
- Scalars are extracted per mesh and stacked into the data array automatically.
- Group label provided per participant.

**Mode B: Raw numpy array** — `add_change_data_directly(data, group_labels)`
- `data`: numpy array, shape `(n_participants, n_vertices)` — one scalar per vertex per participant
- `group_labels`: array of length `n_participants` — integer or string labels (e.g., `[0, 0, 1, 1, 0, 1, ...]`)
- Reference mesh must be set separately via `add_reference_change_mesh()`. Only vertex count is validated (no coordinate check possible since no per-participant meshes are loaded).

**Mode C: Registration path (future, not in scope)** — `add_change_filepath(filepath, group_label=...)`
- Add one mesh file per participant. Meshes need NOT be in correspondence — `compile_data()` handles ICP + pyfocusr registration.
- Not implemented initially; placeholder for when pyfocusr support is added.

**Common to all modes:**
- `reference_mesh`: VTK polydata mesh for clustering and area computation
- Standard SPM parameters: threshold, n_iterations, significance level, etc.

### Outputs

- Vertex-wise t-statistic map (on reference mesh)
- Clustered statistic map with cluster areas
- Null distribution of max cluster sizes
- Threshold cluster size at chosen significance level
- Significant clusters (those exceeding threshold)

## Implementation Plan (Phase 1: Two-Sample T-Test)

### Step 1. `test_statistics.py` — add `IndependentSamplesTTest`

- Inherits from `TestStatistics`
- Constructor accepts additional `group_labels` parameter (the base class `TestStatistics.__init__` takes `data, mesh, return_new_mesh, idx_not_to_include` — call `super().__init__()` with those, then store `self.group_labels = group_labels`)
- Input: `data` shape `(n_participants, n_vertices)`, plus `group_labels` array of length `n_participants`
- `compute_statistics_per_node()`:
  - Split data by group label using `np.unique(self.group_labels)` to get group indices
  - At each vertex: compute group means, pooled SE, t-statistic with df = n_A + n_B - 2
  - Zero out vertices in `idx_not_to_include`
  - Replace non-finite values
  - Store t-statistic as VTK scalars named `'t_statistic'` on the mesh

### Step 2. `permutation_test.py` — add `PermutationGroupLabels`

Named `PermutationGroupLabels` (not `PermutationTwoSampleTTest`) from the start, since the logic — shuffling group labels — is identical for 2-group t-tests and k-group ANOVA. Both use this single class.

**Does not inherit from `PermutationTest`.** The existing base class assumes data-level permutation (takes vertex data, slices by `idx_to_include`, exposes permuted data arrays). Label shuffling is a different paradigm — it only needs `group_labels` and `n_permutations`, not vertex data. A standalone class is cleaner than forcing unused data through the base.

- Constructor: `__init__(self, group_labels, n_permutations)`
  - `self.group_labels = np.asarray(group_labels)`
  - `self.n_participants = len(group_labels)`
  - `self.n_permutations = n_permutations`
  - Compute `self.n_possible_unique_permutations` as `C(n, n_A)` for 2 groups, or `n! / (n_1! * n_2! * ... * n_k!)` (multinomial coefficient) for k groups. Use `math.comb` or `scipy.special.comb` / manual factorial.
  - `self.shuffled_labels = None`

- `perform_permutations()`:
  - If `n_possible_unique_permutations <= n_permutations`: generate all unique permutations (via `itertools` or similar)
  - Else: generate `n_permutations` random shuffles of `group_labels` (using `np.random.permutation`)
  - Store as `self.shuffled_labels`, shape `(n_permutations, n_participants)`

- `get_shuffled_labels()`: returns `self.shuffled_labels`

- `update()`: calls `perform_permutations()`

### Step 3. `monte_carlo.py` — add `MonteCarloClusterGroupTest` + `MonteCarloClusterTwoSampleTest`

**`MonteCarloClusterGroupTest`** — a shared intermediate class for both two-sample and ANOVA MC simulation (Phase 2 adds `MonteCarloClusterANOVA` as a thin subclass). Their loops are structurally identical: pass original data + shuffled labels to a test statistic class. The only differences are which test statistic class to use and the clustering mode.

**Inherits from `MonteCarloSingleTest`.** Accepts `group_labels` as an additional constructor parameter (store as `self.group_labels`). Note: `MonteCarloThreshold.__init__` does not use `**kwargs`, so `group_labels` must be handled explicitly in this subclass's `__init__` before calling `super()`.

- Constructor: `__init__(self, *args, group_labels, test_statistic_class, threshold_type='two_sided', **kwargs)`
  - `self.group_labels = group_labels`
  - `self.test_statistic_class = test_statistic_class` — the class to instantiate per iteration (e.g., `IndependentSamplesTTest`)
  - `self.threshold_type = threshold_type`
  - `self.shuffled_labels = None`

- `update_permutations()`:
  - Instantiate `PermutationGroupLabels(group_labels=self.group_labels, n_permutations=self.n_iterations)`
  - Call `.update()`
  - Store result in `self.shuffled_labels = perm.get_shuffled_labels()` (NOT `self.rand_permutations` — different name to reflect different semantics)
  - Update `self.n_iterations = self.shuffled_labels.shape[0]` (in case all unique permutations < requested)

- `perform_monte_carlo()`:
  - **Different from existing MC classes.** The existing one-sample/correlation MC classes reconstruct a `change_array` from pre-permuted data and handle `idx_to_include` padding (monte_carlo.py:124-130). This class instead passes the **original data** and **shuffled labels** directly to the test statistic class each iteration. No data permutation or `idx_to_include` reconstruction needed.
  - For each iteration:
    - Instantiate `self.test_statistic_class(data=self.data, mesh=self.ref_mesh, return_new_mesh=True, idx_not_to_include=self.idx_not_to_include, group_labels=self.shuffled_labels[iter_])`
    - Call `.compute_statistics_per_node()`
    - Get the statistics mesh, set active scalars to `self.map_name`
    - Record min/max/abs_max test statistics
    - Run `Cluster(statistic_mesh=..., statistic_threshold=self.threshold, threshold_type=self.threshold_type, ...)`
    - Record cluster areas

**`MonteCarloClusterTwoSampleTest`** — thin subclass that sets defaults: `map_name='t_statistic'`, `test_statistic_class=IndependentSamplesTTest`, `threshold_type='two_sided'`. Follows the same constructor pattern as existing subclasses (e.g., `MonteCarloClusterOneSampleTest`).

### Step 4. `knee_spm.py` — add `GroupComparison` + `TwoSampleComparison`

**`GroupComparison`** — intermediate class inheriting from `SingleStatisticSPM`. Holds the shared logic for any group-label-based test:

- Constructor: `__init__(self, *args, group_labels=None, **kwargs)`
  - Call `super().__init__(*args, **kwargs)`
  - `self.group_labels_list = []` — built up incrementally via `add_prealigned_filepath` or `add_change_filepath`
  - `self.group_labels = np.asarray(group_labels) if group_labels is not None else None` — set all at once via constructor or `add_change_data_directly`
  - `self._prealigned_scalars = []` — built up by `add_prealigned_filepath`

- `add_change_data_directly(self, change_data, group_labels)` — **overrides** the base class method (`SingleStatisticSPM.add_change_data_directly(self, change_data)` at knee_spm.py:177). Calls `super().add_change_data_directly(change_data)` then stores `self.group_labels = np.asarray(group_labels)`. Validates:
  - `len(group_labels) == change_data.shape[0]` (one label per participant)
  - `change_data.shape[1] == reference_mesh.GetNumberOfPoints()` if reference mesh is already set (vertex count match)
  - No per-mesh vertex coordinate validation possible here — the user passes a raw numpy array, not meshes. Vertex correspondence is a user guarantee in this mode.

- `add_prealigned_filepath(self, filepath, group_label, scalar_name=None)` — **new method**, the primary input mode for pre-aligned meshes. This is the third input mode alongside `add_change_data_directly` (raw array) and `add_change_filepath` (registration path). Designed for the common case where meshes are already in correspondence (same vertex positions) and the user wants per-mesh validation.
  - Loads the mesh via `read_vtk(filepath)`
  - If this is the first mesh and no reference mesh is set: stores it as the reference mesh (vertex positions become the ground truth)
  - Validates vertex coordinates against the reference mesh: `np.allclose(new_coords, ref_coords, atol=1e-6)`. Raises with a clear error if coordinates don't match — the meshes aren't actually in correspondence.
  - Extracts the scalar array from the mesh. If `scalar_name` is provided, uses `mesh.GetPointData().GetArray(scalar_name)`. If None, uses the active scalars (`GetPointData().GetScalars()`).
  - Appends the scalar array (length `n_vertices`) to `self._prealigned_scalars`
  - Appends `group_label` to `self.group_labels_list`
  - Sets `self.added_change_values_directly = True` (so `update()` skips `compile_data()`)

- `_finalize_prealigned_data()` — stacks `self._prealigned_scalars` into `self.change_values = np.array(self._prealigned_scalars)` shape `(n_participants, n_vertices)` and converts `self.group_labels = np.asarray(self.group_labels_list)`. Only runs if `self._prealigned_scalars` is non-empty; no-ops otherwise.

- `update()` — **overrides** `SingleStatisticSPM.update()`. Calls `self._finalize_prealigned_data()` first, then calls `super().update()`. This is the definitive call site for finalization — it runs before `compile_data()` or `compute_test_statistics()`.

- `add_change_filepath(self, *args, group_label=None, **kwargs)` — **overrides** base class, following the same pattern as `SimpleCorrelation.add_change_filepath` (knee_spm.py:336-354). Calls `super().add_change_filepath(*args, **kwargs)` then appends `group_label` to `self.group_labels_list`. Raises if `group_label` is None. This is for the registration path (pyfocusr) — meshes are NOT expected to be in correspondence yet.

- `add_pre_post_filepaths(self, *args, group_label=None, **kwargs)` — same override pattern. Calls super, appends `group_label`.

- `compile_data()`: left as inherited placeholder for now (registration path not implemented initially). Pre-aligned data via `add_change_data_directly()` or `add_prealigned_filepath()` skips `compile_data()` entirely — `update()` checks `self.added_change_values_directly` (knee_spm.py:216). When registration support is added later, it would mirror `SimpleTimeDifference.compile_data()` and convert `self.group_labels_list` to `self.group_labels = np.asarray(self.group_labels_list)`.

- `_validate_group_labels()` — called at the start of `compute_test_statistics()`:
  - Asserts `self.group_labels is not None`
  - Asserts `len(self.group_labels) == self.change_values.shape[0]`

**`TwoSampleComparison(GroupComparison)`**:
- Defaults `map_name='t_statistic'`
- `compute_test_statistics()`: calls `_validate_group_labels()`, validates `len(np.unique(self.group_labels)) == 2`, instantiates `IndependentSamplesTTest` with `self.change_values`, `self.reference_mesh['mesh']`, and `self.group_labels`
- `compute_mc_thresholds()`: instantiates `MonteCarloClusterTwoSampleTest` with `group_labels=self.group_labels`

### Step 5. `__init__.py` — export new classes

Add `TwoSampleComparison` to imports:
```python
from .knee_spm import SimpleTimeDifference, SimpleCorrelation, TwoSampleComparison
```

(Phase 2 adds `ANOVAComparison` here.)

## Test Plan

Tests go in `tests/test_two_sample.py`. All tests use synthetic data — no real mesh files needed. Create VTK meshes programmatically (e.g., a simple plane or sphere with known vertex positions).

### Test helpers

- `make_simple_mesh(n_points)` — creates a minimal VTK polydata mesh (e.g., a flat triangulated grid) with `n_points` vertices. Returns the mesh. Used by most tests below.
- `make_mesh_with_scalars(n_points, scalar_values, scalar_name)` — same as above but with a scalar array attached. Used by `add_prealigned_filepath` tests (write to a temp .vtk file).

### Input validation tests

**`add_prealigned_filepath`:**
- Test that first mesh sets the reference mesh when none is set
- Test that a second mesh with matching vertex coordinates is accepted
- Test that a second mesh with different vertex coordinates raises an error (the coordinate validation)
- Test that a mesh with different vertex count raises an error
- Test scalar extraction by name (`scalar_name='thickness'`) and by active scalars (`scalar_name=None`)
- Test that `group_label` is stored correctly in `self.group_labels_list`

**`add_change_data_directly`:**
- Test that mismatched `len(group_labels) != data.shape[0]` raises
- Test that mismatched `data.shape[1] != reference_mesh.GetNumberOfPoints()` raises (when reference mesh is set)
- Test that valid data + labels are stored correctly

**`_finalize_prealigned_data`:**
- Test that after adding N participants via `add_prealigned_filepath`, `_finalize_prealigned_data()` produces `change_values` with shape `(N, n_vertices)` and `group_labels` with length N
- Test that calling it when `_prealigned_scalars` is empty is a no-op

**`_validate_group_labels`:**
- Test that it raises when `group_labels` is None
- Test that it raises when label count doesn't match participant count

### `IndependentSamplesTTest` tests

- **Known values:** Create data where group A vertices are all 10.0 and group B vertices are all 0.0 (with some noise). Verify the t-statistic is positive and large at those vertices.
- **Zero difference:** Both groups have the same mean — verify t-statistics are near zero.
- **Verify against scipy:** For a small mesh (e.g., 5 vertices, 10 participants), compute the t-statistic manually with `scipy.stats.ttest_ind` at each vertex and compare to `IndependentSamplesTTest` output.
- **`idx_not_to_include`:** Verify that excluded vertices have t-statistic = 0.
- **Non-finite handling:** Feed data that produces inf/nan (e.g., zero variance in one group) — verify replaced with 0.

### `PermutationGroupLabels` tests

- **Output shape:** `shuffled_labels` has shape `(n_permutations, n_participants)`.
- **Group sizes preserved:** Every row of `shuffled_labels` has the same count of each label as the original.
- **All unique when exhaustive:** For small n (e.g., 4 participants, 2 per group → C(4,2)=6 unique), request more permutations than exist. Verify we get exactly 6 unique rows.
- **Unique permutation count:** Verify `n_possible_unique_permutations` is correct for 2-group and 3-group cases.

### `MonteCarloClusterGroupTest` / `MonteCarloClusterTwoSampleTest` tests

- **Smoke test:** Run with small data (e.g., 10 participants, small mesh, 50 iterations). Verify it completes and produces a non-empty cluster size distribution.
- **Null distribution shape:** `get_distribution_of_max_clustersizes()` returns a list of length `n_iterations`.
- **Threshold is non-negative:** `get_threshold_clustersize()` returns a value >= 0.
- **Uses shuffled labels, not rand_permutations:** Verify `self.shuffled_labels` is populated and `self.rand_permutations` is None (the attribute from the parent class).

### `TwoSampleComparison` end-to-end tests

- **Mode A (pre-aligned files):** Create temp VTK files with known scalars. Add via `add_prealigned_filepath`. Call `update()`. Verify test statistic map exists on the reference mesh.
- **Mode B (raw array):** Pass numpy array via `add_change_data_directly`. Call `update()`. Verify same outputs.
- **Group count validation:** Passing 1 group or 3 groups to `TwoSampleComparison` should raise.
- **Known effect:** Create data with a clear group difference at specific vertices and no difference elsewhere. Run full pipeline. Verify that clusters appear at the expected vertices and that the cluster size exceeds the MC threshold (i.e., the effect is detected as significant).
- **No effect (null):** Create data with no group difference. Run full pipeline. Verify no significant clusters (or very few — permutation test should not reject the null).

## ANOVA Extension (Phase 2, 3+ Groups)

### Why this is straightforward

The t-test and ANOVA share the same permutation scheme (shuffle participant group labels) and the same pipeline (compute statistic → threshold → cluster → compare to null distribution). The only difference is the vertex-wise statistic: F instead of t, and one-sided clustering.

This is why the Phase 1 implementation uses shared intermediate classes (`PermutationGroupLabels`, `MonteCarloClusterGroupTest`, `GroupComparison`) — the ANOVA classes are thin wrappers that swap in the F-statistic and one-sided thresholding.

### What to implement in Phase 2

1. `test_statistics.py` — add `OneWayANOVA` (inherits `TestStatistics`). Input: `data` shape `(n_participants, n_vertices)` + `group_labels`. Computes F per vertex using vectorized SSB/SSW. Stores F-statistic as VTK scalars named `'f_statistic'`.

2. `monte_carlo.py` — add `MonteCarloClusterANOVA`. Thin subclass of `MonteCarloClusterGroupTest`: defaults `map_name='f_statistic'`, `test_statistic_class=OneWayANOVA`, `threshold_type='one_sided'`.

3. `knee_spm.py` — add `ANOVAComparison(GroupComparison)`:
   - Defaults `map_name='f_statistic'`
   - `compute_test_statistics()`: calls `_validate_group_labels()`, validates `len(np.unique(self.group_labels)) >= 2`, instantiates `OneWayANOVA`
   - `compute_mc_thresholds()`: instantiates `MonteCarloClusterANOVA` with `group_labels=self.group_labels`
   - Overrides `cluster_test_statistics()` to use `threshold_type='one_sided'` (F is always positive)

4. `__init__.py` — add `ANOVAComparison` to exports.

### F-statistic computation (fully vectorized)

The one-way ANOVA F-statistic is computed per vertex using numpy vector operations. The only loop is over groups (e.g., 3 iterations for 3 groups), which is trivial. All vertex-wise operations are vectorized:

```
grand_mean = mean across all participants          # (n_vertices,)
for each group k:
    group_mean_k = mean of group k                 # (n_vertices,)
    SSB += n_k * (group_mean_k - grand_mean)^2     # vectorized over vertices
    SSW += sum((x_ij - group_mean_k)^2) for i in k # vectorized over vertices
F = (SSB / (k-1)) / (SSW / (n-k))                  # vectorized over vertices
```

### Permutation scheme

Identical to the two-sample case — `PermutationGroupLabels` shuffles group labels regardless of number of groups. The unique permutation count uses the multinomial coefficient `n! / (n_1! * n_2! * ... * n_k!)` which reduces to `C(n, n_A)` for k=2.

Note: the F-statistic is one-sided (always positive), so cluster thresholding uses `one_sided` mode in `Cluster` rather than `two_sided`.

### Practical use for 3-group comparison

- Run `ANOVAComparison` as an omnibus test to find vertices/clusters where any group differs.
- Follow up with pairwise `TwoSampleComparison` (A vs B, A vs C, B vs C) to identify which groups differ where.

## Future-proofing: naming conventions for new test types

The naming pattern established here should be followed for any future group-based tests:

- **Permutation classes**: if the permutation scheme is "shuffle participant labels/assignments," reuse `PermutationGroupLabels`. Only create a new permutation class if the scheme is fundamentally different (e.g., sign-flip, within-subject shuffle).
- **MC classes**: if the MC loop is "pass original data + permuted labels to a test statistic class," inherit from `MonteCarloClusterGroupTest` and just set the test statistic class + threshold type. Only create a new MC base if the loop structure differs (e.g., the existing one-sample/correlation pattern of pre-permuting data arrays).
- **SPM classes**: if the test uses group labels, inherit from `GroupComparison`. If not, inherit from `SingleStatisticSPM` directly.
- **Attribute naming**: `self.shuffled_labels` for permuted group assignments (shape `n_permutations × n_participants`). `self.rand_permutations` / `self.rand_sign_permutations` remain for the existing data-permutation pattern. Do not mix these.

## Notes

- Both tests use the same `group_labels` interface — pass any array of labels, the code splits by unique values. The t-test is just the k=2 special case of the same design.
- The permutation class is shared between t-test and ANOVA since both just shuffle participant group labels.
- This sits naturally alongside the existing one-sample test (sign-flip permutations) and correlation test (label-shuffle on two variables).
