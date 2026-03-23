# Plan: Group Comparison Tests (Two-Sample T-Test & One-Way ANOVA)

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

### Inputs

- `data`: numpy array, shape `(n_participants, n_vertices)` — one scalar per vertex per participant
- `group_labels`: array of length `n_participants` — integer or string labels indicating group membership (e.g., `[0, 0, 1, 1, 0, 1, ...]`)
- `reference_mesh`: VTK polydata mesh for clustering and area computation
- Standard SPM parameters: threshold, n_iterations, significance level, etc.

### Outputs

- Vertex-wise t-statistic map (on reference mesh)
- Clustered statistic map with cluster areas
- Null distribution of max cluster sizes
- Threshold cluster size at chosen significance level
- Significant clusters (those exceeding threshold)

## Implementation Plan

### 1. `test_statistics.py` — add `IndependentSamplesTTest`

- Inherits from `TestStatistics`
- Input: `data` shape `(n_participants, n_vertices)`, plus `group_labels` array
- `compute_statistics_per_node()`:
  - Split data by group label
  - At each vertex: compute means, pooled SE, t-statistic with appropriate df
  - Zero out vertices in `idx_not_to_include`
  - Replace non-finite values
  - Store t-statistic as VTK scalars named `'t_statistic'` on the mesh

### 2. `permutation_test.py` — add `PermutationTwoSampleTTest`

- Inherits from `PermutationTest`
- Input: `data` shape `(n_participants, n_vertices)`, `group_labels`, `n_permutations`
- `perform_permutations()`: generate N random shuffles of `group_labels` (or all unique shuffles if fewer than N exist). Each shuffle reassigns whole participants to groups — per-vertex data stays intact. Store as a matrix of shape `(n_permutations, n_participants)` of shuffled label arrays.
- No need to rearrange the actual vertex data — just shuffle which group each participant belongs to, then pass the shuffled labels to the t-test.

### 3. `monte_carlo.py` — add `MonteCarloClusterTwoSampleTest`

- Inherits from `MonteCarloSingleTest`
- Input: same as `MonteCarloThreshold` plus `group_labels`
- `update_permutations()`: uses `PermutationTwoSampleTTest` to generate shuffled label sets
- `perform_monte_carlo()`: for each permutation iteration:
  - Run `IndependentSamplesTTest` with the shuffled labels
  - Threshold, cluster, record max cluster size and max test statistic
  - Same pattern as existing `MonteCarloClusterOneSampleTest`

### 4. `knee_spm.py` — add `TwoSampleComparison`

- Inherits from `SingleStatisticSPM`
- Constructor takes `group_labels` (or accepts them via a method)
- `add_change_data_directly(data, group_labels)` — accepts pre-aligned data + group membership
- `compute_test_statistics()`: instantiates `IndependentSamplesTTest` with data and labels
- `compute_mc_thresholds()`: instantiates `MonteCarloClusterTwoSampleTest`
- `compile_data()`: optional pyfocusr registration path (lazy import), same as existing classes
- `update()` inherited from `SingleStatisticSPM` — runs the full pipeline

### 5. `__init__.py` — export `TwoSampleComparison`

Add to the imports so it's accessible as `pyKneeSPM.TwoSampleComparison`.

## ANOVA Extension (3+ Groups)

### Why this is straightforward

The t-test and ANOVA share the same permutation scheme (shuffle participant group labels) and the same pipeline (compute statistic → threshold → cluster → compare to null distribution). The only difference is the vertex-wise statistic: F instead of t.

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

Identical to the two-sample case — shuffle which group each participant belongs to, recompute F at every vertex, threshold, cluster, record max cluster size. The null distribution is built the same way.

Note: the F-statistic is one-sided (always positive), so cluster thresholding uses `one_sided` mode in `Cluster` rather than `two_sided`.

### Implementation — mirrors the t-test classes

1. `test_statistics.py` — add `OneWayANOVA` (inherits `TestStatistics`). Input: `data` shape `(n_participants, n_vertices)` + `group_labels`. Computes F per vertex using vectorized SSB/SSW. Stores F-statistic as VTK scalars named `'f_statistic'`.

2. `permutation_test.py` — **reuse `PermutationTwoSampleTTest`** (rename to `PermutationGroupLabels` or similar). It just shuffles group labels — works for any number of groups.

3. `monte_carlo.py` — add `MonteCarloClusterANOVA`. Same as `MonteCarloClusterTwoSampleTest` but uses `OneWayANOVA` instead of `IndependentSamplesTTest`, and uses `one_sided` clustering.

4. `knee_spm.py` — add `ANOVAComparison` (or generalize `TwoSampleComparison` to handle both, since the only difference is which test statistic class to instantiate).

### Practical use for 3-group comparison

- Run `ANOVAComparison` as an omnibus test to find vertices/clusters where any group differs.
- Follow up with pairwise `TwoSampleComparison` (A vs B, A vs C, B vs C) to identify which groups differ where.

## Notes

- Both tests use the same `group_labels` interface — pass any array of labels, the code splits by unique values. The t-test is just the k=2 special case of the same design.
- The permutation class can be shared between t-test and ANOVA since both just shuffle participant group labels.
- This sits naturally alongside the existing one-sample test (sign-flip permutations) and correlation test (label-shuffle on two variables).
