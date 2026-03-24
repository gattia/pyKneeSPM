# PyKneeSPM

Python module to perform SPM (statistical parametric mapping) style analysis on 3D surface meshes of knee bones.

## Introduction

This work is inspired by the brain imaging community's use of SPM to find "activations" in the brain and the lack of
ways to find anatomically specific information about musculoskeletal joint health. Current methods use human defined
ROIs that may or may not include the most relevant information. This package finds clusters of significant change
directly on the bone surface.

This package works under the premise that knee information can be localized to the surface of the bone. For example,
we can easily assign cartilage thickness or T2 values to the bone surface for statistical analysis. We then register
the bones to a common reference frame to enable calculation of group statistics.

## Statistical Tests

- **One-sample z-test** (`SimpleTimeDifference`) — test whether change values differ from zero across participants
- **Correlation t-test** (`SimpleCorrelation`) — test the correlation between a surface scalar and a secondary variable
- **Two-sample t-test** (`TwoSampleComparison`) — compare a surface scalar between two groups of participants
- **One-way ANOVA** (`ANOVAComparison`) — omnibus test for differences across 2+ groups

All tests use permutation-based cluster-extent inference to control for multiple comparisons across vertices.

## Installation

```bash
git clone <repo_url>
cd pykneespm
pip install .
```

## Quick Start: Two-Sample Comparison

### From pre-aligned mesh files

If your meshes are already registered (same vertex positions across participants), you can load them directly. Vertex coordinates are validated against the reference on each add.

```python
import pyKneeSPM

spm = pyKneeSPM.TwoSampleComparison(
    map_threshold=2.33,
    n_monte_carlo_iterations=10000,
)

# Add each participant's mesh — coordinates are checked against the first mesh
for filepath, group in participant_data:
    spm.add_prealigned_filepath(filepath, group_label=group, scalar_name='thickness')

spm.update()

# Results
sig_clusters = spm.get_significant_clusters()
threshold = spm.get_threshold_cluster_size()
t_map = spm.get_test_statistic_maps()
```

### From a numpy array

If you've already extracted scalars into a matrix:

```python
import numpy as np
import pyKneeSPM

# data shape: (n_participants, n_vertices)
# group_labels: array of 0s and 1s, length n_participants
spm = pyKneeSPM.TwoSampleComparison(
    map_threshold=2.33,
    n_monte_carlo_iterations=10000,
)
spm.add_reference_change_mesh('path/to/reference.vtk')
spm.add_change_data_directly(data, group_labels)
spm.update()

sig_clusters = spm.get_significant_clusters()
```

### ANOVA (3+ groups)

Use `ANOVAComparison` as an omnibus test, then follow up with pairwise `TwoSampleComparison` to identify which groups differ.

```python
spm = pyKneeSPM.ANOVAComparison(
    map_threshold=3.0,
    n_monte_carlo_iterations=10000,
)

for filepath, group in participant_data:  # group labels: 0, 1, 2, ...
    spm.add_prealigned_filepath(filepath, group_label=group, scalar_name='thickness')

spm.update()

sig_clusters = spm.get_significant_clusters()
```

### One-sample test (change from baseline)

```python
spm = pyKneeSPM.SimpleTimeDifference(
    map_threshold=2.33,
    n_monte_carlo_iterations=10000,
)
spm.add_reference_change_mesh('path/to/reference.vtk')
spm.add_change_data_directly(change_data)  # shape: (n_participants, n_vertices)
spm.update()
```

## How It Works

1. **Vertex-wise test statistics** — compute a test statistic (z, t, F, or correlation) at every vertex on the surface mesh
2. **Cluster extraction** — threshold the statistic map and extract contiguous clusters using VTK connectivity filters
3. **Permutation inference** — build a null distribution of maximum cluster sizes by shuffling data/labels and repeating steps 1-2 thousands of times. Clusters larger than the 95th percentile of this null distribution are significant (family-wise error corrected)

## Dependencies

Core: `vtk`, `numpy`, `scipy`, `sympy`

Optional (registration only): `pyfocusr` — only needed if using `compile_data()` for mesh registration. Not required for pre-aligned data.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

MIT License.
