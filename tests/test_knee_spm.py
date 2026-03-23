import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from pyKneeSPM.knee_spm import SimpleTimeDifference, SimpleCorrelation


def _get_cluster_point_coords(cluster_mesh):
    """Extract point coordinates from a VTK cluster mesh as an (N, 3) array."""
    n = cluster_mesh.GetNumberOfPoints()
    return np.array([cluster_mesh.GetPoint(i) for i in range(n)])


def _cluster_overlaps_region(cluster_mesh, reference_mesh, region_idx, tol=1.1):
    """Check whether a cluster's points fall within the expected region.

    VTK's banded contour filter interpolates new points along contour
    boundaries, so cluster coordinates won't exactly match original vertices.
    We use tol=1.1 (just over one grid spacing) to account for boundary
    interpolation by the contour filter.

    Returns the fraction of cluster points that are within `tol` of a
    vertex in `region_idx`.
    """
    cluster_pts = _get_cluster_point_coords(cluster_mesh)
    region_pts = np.array([reference_mesh.GetPoint(i) for i in region_idx])
    overlapping = 0
    for cp in cluster_pts:
        dists = np.linalg.norm(region_pts - cp, axis=1)
        if np.min(dists) < tol:
            overlapping += 1
    return overlapping / len(cluster_pts) if len(cluster_pts) > 0 else 0.0


@pytest.mark.slow
class TestSimpleTimeDifferenceIntegration:
    def test_full_pipeline_with_direct_data(self, small_mesh):
        """Run the full pipeline with pre-aligned synthetic data."""
        np.random.seed(42)
        n_participants = 15
        n_vertices = small_mesh.GetNumberOfPoints()

        # Create data with a strong localized effect
        data = np.random.randn(n_participants, n_vertices) * 0.5
        # Add strong signal to first 10 vertices
        data[:, :10] += 5.0

        spm = SimpleTimeDifference(
            map_threshold=2.33,
            n_monte_carlo_iterations=200,
            compute_min_cluster_size=True,
        )
        spm.reference_mesh = {'mesh': small_mesh}
        spm.add_change_data_directly(data)
        spm.update()

        # Test statistic map should exist
        stat_maps = spm.get_test_statistic_maps()
        assert 'z_statistic' in stat_maps
        z_mesh = stat_maps['z_statistic']
        assert z_mesh.GetNumberOfPoints() == n_vertices

        # Clustered maps should exist
        clustered = spm.get_clustered_statistic_maps()
        assert 'z_statistic' in clustered
        assert 2.33 in clustered['z_statistic']

        # MC thresholds should have been computed
        threshold_size = spm.get_threshold_cluster_size()
        assert 2.33 in threshold_size

    def test_detects_significant_change(self, flat_grid_mesh):
        """Synthetic data with a strong localized effect should be detected as significant."""
        np.random.seed(42)
        n_participants = 20
        n_vertices = flat_grid_mesh.GetNumberOfPoints()  # 100 points (10x10 grid)

        # Pure noise baseline
        data = np.random.randn(n_participants, n_vertices) * 0.5

        # Inject a large positive shift on a spatial cluster (top-left quadrant: x<5, y>=5)
        effect_idx = []
        for i in range(n_vertices):
            x, y, _ = flat_grid_mesh.GetPoint(i)
            if x < 5 and y >= 5:
                effect_idx.append(i)
        data[:, effect_idx] += 8.0

        spm = SimpleTimeDifference(
            map_threshold=2.33,
            n_monte_carlo_iterations=100,
            compute_min_cluster_size=True,
        )
        spm.reference_mesh = {'mesh': flat_grid_mesh}
        spm.add_change_data_directly(data)
        spm.update()

        # Verify the z-scores are large in the effect region
        z_mesh = spm.get_test_statistic_maps()['z_statistic']
        z_vals = vtk_to_numpy(z_mesh.GetPointData().GetArray('z_statistic'))
        assert np.all(z_vals[effect_idx] > 2.33), "Effect region z-scores should exceed threshold"

        # Verify at least one cluster exceeds the MC threshold (i.e., is significant)
        threshold_size = spm.get_threshold_cluster_size()[2.33]
        clustered = spm.get_clustered_statistic_maps()['z_statistic'][2.33]
        cluster_areas = [clustered[k]['area'] for k in clustered]
        significant_clusters = [a for a in cluster_areas if a >= threshold_size]
        assert len(significant_clusters) >= 1, "Should detect at least one significant cluster"

        # Verify the significant cluster(s) actually overlap the injected effect region
        for k in clustered:
            if clustered[k]['area'] >= threshold_size:
                overlap = _cluster_overlaps_region(
                    clustered[k]['mesh'], flat_grid_mesh, effect_idx
                )
                assert overlap > 0.5, (
                    f"Significant cluster should overlap effect region, "
                    f"but only {overlap:.0%} of its points do"
                )

    def test_noise_produces_no_significant_clusters(self, flat_grid_mesh):
        """Pure noise data should rarely produce significant clusters.

        Runs multiple seeds and checks that the false-positive rate stays
        within the expected FWE-controlled bound (at most 1 out of 5 runs).
        """
        n_participants = 20
        n_vertices = flat_grid_mesh.GetNumberOfPoints()
        false_positives = 0
        n_runs = 5

        for seed in range(n_runs):
            np.random.seed(seed + 100)
            data = np.random.randn(n_participants, n_vertices)

            spm = SimpleTimeDifference(
                map_threshold=2.33,
                n_monte_carlo_iterations=500,
                compute_min_cluster_size=True,
            )
            spm.reference_mesh = {'mesh': flat_grid_mesh}
            spm.add_change_data_directly(data)
            spm.update()

            threshold_size = spm.get_threshold_cluster_size()[2.33]
            clustered = spm.get_clustered_statistic_maps()['z_statistic'][2.33]
            cluster_areas = [clustered[k]['area'] for k in clustered]
            significant_clusters = [a for a in cluster_areas if a >= threshold_size]
            false_positives += len(significant_clusters) > 0

        assert false_positives <= 1, (
            f"Too many false positives: {false_positives}/{n_runs} noise runs "
            f"produced significant clusters (expected <= 1)"
        )

    def test_no_mc_when_disabled(self, small_mesh):
        """When compute_min_cluster_size=False, MC thresholds should not be computed."""
        np.random.seed(42)
        n_participants = 10
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.randn(n_participants, n_vertices)

        spm = SimpleTimeDifference(
            map_threshold=2.33,
            compute_min_cluster_size=False,
        )
        spm.reference_mesh = {'mesh': small_mesh}
        spm.add_change_data_directly(data)
        spm.update()

        threshold_size = spm.get_threshold_cluster_size()
        assert len(threshold_size) == 0


@pytest.mark.slow
class TestSimpleCorrelationIntegration:
    def test_full_pipeline_with_direct_data(self, small_mesh):
        """Run the full correlation pipeline with pre-aligned synthetic data."""
        np.random.seed(42)
        n_participants = 15
        n_vertices = small_mesh.GetNumberOfPoints()

        # Create correlated data
        data = np.zeros((n_participants, n_vertices, 2))
        data[:, :, 0] = np.random.randn(n_participants, n_vertices)
        # Strong correlation on first 10 vertices
        data[:, :10, 1] = data[:, :10, 0] * 3.0 + np.random.randn(n_participants, 10) * 0.1
        # No correlation on remaining vertices
        data[:, 10:, 1] = np.random.randn(n_participants, n_vertices - 10)

        spm = SimpleCorrelation(
            map_threshold=2.33,
            n_monte_carlo_iterations=200,
            compute_min_cluster_size=True,
        )
        spm.reference_mesh = {'mesh': small_mesh}
        spm.add_change_data_directly(data)
        spm.update()

        # Test statistic map should exist
        stat_maps = spm.get_test_statistic_maps()
        assert 't_statistic' in stat_maps
        t_mesh = stat_maps['t_statistic']
        assert t_mesh.GetNumberOfPoints() == n_vertices

        # Clustered maps should exist
        clustered = spm.get_clustered_statistic_maps()
        assert 't_statistic' in clustered

        # MC thresholds should have been computed
        threshold_size = spm.get_threshold_cluster_size()
        assert 2.33 in threshold_size

    def test_detects_significant_correlation(self, flat_grid_mesh):
        """Synthetic data with a strong localized correlation should be detected as significant."""
        np.random.seed(42)
        n_participants = 25
        n_vertices = flat_grid_mesh.GetNumberOfPoints()  # 100 points

        data = np.zeros((n_participants, n_vertices, 2))
        data[:, :, 0] = np.random.randn(n_participants, n_vertices)
        data[:, :, 1] = np.random.randn(n_participants, n_vertices)  # noise baseline

        # Inject strong correlation on top-left quadrant (x<5, y>=5)
        effect_idx = []
        for i in range(n_vertices):
            x, y, _ = flat_grid_mesh.GetPoint(i)
            if x < 5 and y >= 5:
                effect_idx.append(i)
        data[:, effect_idx, 1] = data[:, effect_idx, 0] * 3.0 + np.random.randn(n_participants, len(effect_idx)) * 0.1

        spm = SimpleCorrelation(
            map_threshold=2.33,
            n_monte_carlo_iterations=100,
            compute_min_cluster_size=True,
        )
        spm.reference_mesh = {'mesh': flat_grid_mesh}
        spm.add_change_data_directly(data)
        spm.update()

        # Verify at least one cluster exceeds the MC threshold
        threshold_size = spm.get_threshold_cluster_size()[2.33]
        clustered = spm.get_clustered_statistic_maps()['t_statistic'][2.33]
        cluster_areas = [clustered[k]['area'] for k in clustered]
        significant_clusters = [a for a in cluster_areas if a >= threshold_size]
        assert len(significant_clusters) >= 1, "Should detect at least one significant correlation cluster"

        # Verify the significant cluster(s) actually overlap the injected effect region
        for k in clustered:
            if clustered[k]['area'] >= threshold_size:
                overlap = _cluster_overlaps_region(
                    clustered[k]['mesh'], flat_grid_mesh, effect_idx
                )
                assert overlap > 0.5, (
                    f"Significant correlation cluster should overlap effect region, "
                    f"but only {overlap:.0%} of its points do"
                )

    def test_noise_produces_no_significant_clusters(self, flat_grid_mesh):
        """Pure noise correlation data should rarely produce significant clusters.

        Runs multiple seeds and checks that the false-positive rate stays
        within the expected FWE-controlled bound (at most 1 out of 5 runs).
        """
        n_participants = 25
        n_vertices = flat_grid_mesh.GetNumberOfPoints()
        false_positives = 0
        n_runs = 5

        for seed in range(n_runs):
            np.random.seed(seed + 200)
            data = np.zeros((n_participants, n_vertices, 2))
            data[:, :, 0] = np.random.randn(n_participants, n_vertices)
            data[:, :, 1] = np.random.randn(n_participants, n_vertices)

            spm = SimpleCorrelation(
                map_threshold=2.33,
                n_monte_carlo_iterations=500,
                compute_min_cluster_size=True,
            )
            spm.reference_mesh = {'mesh': flat_grid_mesh}
            spm.add_change_data_directly(data)
            spm.update()

            threshold_size = spm.get_threshold_cluster_size()[2.33]
            clustered = spm.get_clustered_statistic_maps()['t_statistic'][2.33]
            cluster_areas = [clustered[k]['area'] for k in clustered]
            significant_clusters = [a for a in cluster_areas if a >= threshold_size]
            false_positives += len(significant_clusters) > 0

        assert false_positives <= 1, (
            f"Too many false positives: {false_positives}/{n_runs} noise runs "
            f"produced significant correlation clusters (expected <= 1)"
        )
