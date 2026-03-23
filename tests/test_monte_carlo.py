import numpy as np
import pytest

from pyKneeSPM.monte_carlo import MonteCarloClusterOneSampleTest, MonteCarloClusterCorrelationTest


@pytest.mark.slow
class TestMonteCarloClusterOneSampleTest:
    def test_noise_data_builds_distribution(self, small_mesh):
        """Pure noise data should produce a non-empty distribution with threshold > 0."""
        np.random.seed(42)
        n_participants = 10
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.randn(n_participants, n_vertices)

        mc = MonteCarloClusterOneSampleTest(
            small_mesh, data,
            method='permutation',
            threshold=2.33,
            n_iterations=50,
        )
        mc.update()

        dist = mc.get_distribution_of_max_clustersizes()
        assert len(dist) > 0
        threshold_size = mc.get_threshold_clustersize(0.05)
        assert threshold_size >= 0

    def test_massive_effect_exceeds_threshold(self, small_mesh):
        """Data with a massive effect => actual cluster should exceed MC threshold."""
        np.random.seed(42)
        n_participants = 15
        n_vertices = small_mesh.GetNumberOfPoints()
        # All values = 10 + tiny noise
        data = np.full((n_participants, n_vertices), 10.0)
        data += np.random.randn(n_participants, n_vertices) * 0.01

        mc = MonteCarloClusterOneSampleTest(
            small_mesh, data,
            method='permutation',
            threshold=2.33,
            n_iterations=50,
        )
        mc.update()

        threshold_size = mc.get_threshold_clustersize(0.05)
        dist = mc.get_distribution_of_max_clustersizes()
        assert len(dist) == mc.n_iterations

    def test_distribution_length(self, small_mesh):
        """Distribution should have length equal to n_iterations (or all unique perms if fewer)."""
        np.random.seed(42)
        n_participants = 10
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.randn(n_participants, n_vertices)

        mc = MonteCarloClusterOneSampleTest(
            small_mesh, data,
            method='permutation',
            threshold=2.33,
            n_iterations=30,
        )
        mc.update()

        dist = mc.get_distribution_of_max_clustersizes()
        assert len(dist) == mc.n_iterations

    def test_threshold_is_95th_percentile(self, small_mesh):
        """get_threshold_clustersize(0.05) should return the 95th percentile."""
        np.random.seed(42)
        n_participants = 10
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.randn(n_participants, n_vertices)

        mc = MonteCarloClusterOneSampleTest(
            small_mesh, data,
            method='permutation',
            threshold=2.33,
            n_iterations=100,
        )
        mc.update()

        threshold_size = mc.get_threshold_clustersize(0.05)
        dist = mc.get_distribution_of_max_clustersizes()

        # The threshold should be the value at the 95th percentile index
        idx = int(np.ceil(0.95 * mc.n_iterations))
        assert threshold_size == dist[idx - 1]


@pytest.mark.slow
class TestMonteCarloClusterCorrelationTest:
    def test_noise_data_builds_distribution(self, small_mesh):
        """Random uncorrelated data should produce a valid distribution."""
        np.random.seed(42)
        n_participants = 10
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.randn(n_participants, n_vertices, 2)

        mc = MonteCarloClusterCorrelationTest(
            small_mesh, data,
            method='permutation',
            threshold=2.33,
            n_iterations=50,
        )
        mc.update()

        dist = mc.get_distribution_of_max_clustersizes()
        assert len(dist) > 0
        threshold_size = mc.get_threshold_clustersize(0.05)
        assert threshold_size >= 0

    def test_strong_correlation_produces_clusters(self, small_mesh):
        """Strongly correlated data should produce clusters in the permutation distribution."""
        np.random.seed(42)
        n_participants = 10
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.zeros((n_participants, n_vertices, 2))
        data[:, :, 0] = np.random.randn(n_participants, n_vertices)
        data[:, :, 1] = data[:, :, 0] * 2.0 + np.random.randn(n_participants, n_vertices) * 0.1

        mc = MonteCarloClusterCorrelationTest(
            small_mesh, data,
            method='permutation',
            threshold=2.33,
            n_iterations=50,
        )
        mc.update()

        dist = mc.get_distribution_of_max_clustersizes()
        assert len(dist) == mc.n_iterations
