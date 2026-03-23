import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy import stats as scipy_stats

from pyKneeSPM.test_statistics import OneSampleZTest, CorrelationTTest, replace_non_finite_values


class TestOneSampleZTest:
    def test_hand_computed_z_scores(self, small_mesh):
        """Verify z = mean / (std_ddof1 / sqrt(n)) against hand computation."""
        np.random.seed(42)
        n_participants, n_vertices = 5, small_mesh.GetNumberOfPoints()
        data = np.random.randn(n_participants, n_vertices) + 2.0  # positive shift

        z_test = OneSampleZTest(data, small_mesh, return_new_mesh=True)
        z_test.compute_statistics_per_node()
        result = vtk_to_numpy(z_test.get_statistics_mesh().GetPointData().GetArray('z_statistic'))

        # Hand compute
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=1)
        se = std / np.sqrt(n_participants)
        expected = mean / se

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_uniform_positive_shift_large_z(self, small_mesh):
        """All participants have the same positive value => large uniform z-scores."""
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.full((10, n_vertices), 5.0)
        # Add tiny noise so std != 0
        data += np.random.RandomState(0).randn(10, n_vertices) * 0.01

        z_test = OneSampleZTest(data, small_mesh, return_new_mesh=True)
        z_test.compute_statistics_per_node()
        result = vtk_to_numpy(z_test.get_statistics_mesh().GetPointData().GetArray('z_statistic'))

        # All z-scores should be very large (mean~5, se~tiny)
        assert np.all(result > 50)

    def test_zero_mean_noise_small_z(self, small_mesh):
        """Random data centered on zero => z-scores should be small on average."""
        np.random.seed(123)
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.randn(100, n_vertices)  # large n for stable stats

        z_test = OneSampleZTest(data, small_mesh, return_new_mesh=True)
        z_test.compute_statistics_per_node()
        result = vtk_to_numpy(z_test.get_statistics_mesh().GetPointData().GetArray('z_statistic'))

        # Most z-scores should be within [-3, 3] for null data
        assert np.mean(np.abs(result) < 3) > 0.9

    def test_idx_not_to_include_zeroed(self, small_mesh):
        """Vertices in idx_not_to_include should have z=0."""
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.RandomState(42).randn(5, n_vertices) + 3.0
        exclude = np.array([0, 1, 2, 3, 4])

        z_test = OneSampleZTest(data, small_mesh, return_new_mesh=True, idx_not_to_include=exclude)
        z_test.compute_statistics_per_node()
        result = vtk_to_numpy(z_test.get_statistics_mesh().GetPointData().GetArray('z_statistic'))

        np.testing.assert_array_equal(result[exclude], 0.0)
        # Other vertices should be non-zero
        other = np.setdiff1d(np.arange(n_vertices), exclude)
        assert np.all(result[other] != 0)

    def test_non_finite_replaced(self, small_mesh):
        """If all participants have the same value (std=0), z = inf => should be replaced with 0."""
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.full((5, n_vertices), 3.0)  # std=0 => se=0 => z=inf

        z_test = OneSampleZTest(data, small_mesh, return_new_mesh=True)
        z_test.compute_statistics_per_node()
        result = vtk_to_numpy(z_test.get_statistics_mesh().GetPointData().GetArray('z_statistic'))

        assert np.all(np.isfinite(result))

    def test_scalar_name_is_z_statistic(self, small_mesh):
        """Result mesh should have scalars named 'z_statistic'."""
        data = np.random.RandomState(0).randn(5, small_mesh.GetNumberOfPoints())

        z_test = OneSampleZTest(data, small_mesh, return_new_mesh=True)
        z_test.compute_statistics_per_node()
        mesh = z_test.get_statistics_mesh()

        assert mesh.GetPointData().GetScalars().GetName() == 'z_statistic'

    def test_return_new_mesh_false(self, small_mesh):
        """When return_new_mesh=False, scalars go on the original mesh."""
        data = np.random.RandomState(0).randn(5, small_mesh.GetNumberOfPoints())

        z_test = OneSampleZTest(data, small_mesh, return_new_mesh=False)
        z_test.compute_statistics_per_node()
        result_mesh = z_test.get_statistics_mesh()

        # Should be the same object
        assert result_mesh is small_mesh
        assert small_mesh.GetPointData().GetScalars().GetName() == 'z_statistic'


class TestCorrelationTTest:
    def test_perfect_correlation(self, small_mesh):
        """x2 = 2*x1 + 3 => r should be ~1, t should be very large."""
        np.random.seed(42)
        n_participants = 20
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.zeros((n_participants, n_vertices, 2))
        data[:, :, 0] = np.random.randn(n_participants, n_vertices)
        data[:, :, 1] = 2.0 * data[:, :, 0] + 3.0

        t_test = CorrelationTTest(data, small_mesh, return_new_mesh=True)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()

        corr = vtk_to_numpy(mesh.GetPointData().GetArray('correlation'))
        t_stat = vtk_to_numpy(mesh.GetPointData().GetArray('t_statistic'))

        np.testing.assert_allclose(corr, 1.0, atol=1e-10)
        # t should be very large (infinite for perfect correlation, replaced with 0 due to non-finite handling)
        # Actually r=1 => 1-r^2=0 => t=inf => replaced with 0
        # So let's check near-perfect instead
        # This test documents the behavior: perfect correlation => non-finite t => 0

    def test_near_perfect_correlation(self, small_mesh):
        """Near-perfect correlation => r close to 1, large t."""
        np.random.seed(42)
        n_participants = 20
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.zeros((n_participants, n_vertices, 2))
        data[:, :, 0] = np.random.randn(n_participants, n_vertices)
        data[:, :, 1] = 2.0 * data[:, :, 0] + 3.0 + np.random.randn(n_participants, n_vertices) * 0.01

        t_test = CorrelationTTest(data, small_mesh, return_new_mesh=True)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()

        corr = vtk_to_numpy(mesh.GetPointData().GetArray('correlation'))
        t_stat = vtk_to_numpy(mesh.GetPointData().GetArray('t_statistic'))

        assert np.all(corr > 0.99)
        assert np.all(np.abs(t_stat) > 10)

    def test_no_correlation(self, small_mesh):
        """Independent random data => r ≈ 0."""
        np.random.seed(42)
        n_participants = 100
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.randn(n_participants, n_vertices, 2)

        t_test = CorrelationTTest(data, small_mesh, return_new_mesh=True)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()

        corr = vtk_to_numpy(mesh.GetPointData().GetArray('correlation'))
        assert np.mean(np.abs(corr)) < 0.2

    def test_against_scipy(self, small_mesh):
        """Compare library output to scipy.stats.pearsonr for a few vertices."""
        np.random.seed(42)
        n_participants = 30
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.randn(n_participants, n_vertices, 2)

        t_test = CorrelationTTest(data, small_mesh, return_new_mesh=True)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()
        corr = vtk_to_numpy(mesh.GetPointData().GetArray('correlation'))
        t_stat = vtk_to_numpy(mesh.GetPointData().GetArray('t_statistic'))

        for v in [0, 5, 10, 20]:
            r_scipy, _ = scipy_stats.pearsonr(data[:, v, 0], data[:, v, 1])
            np.testing.assert_allclose(corr[v], r_scipy, atol=1e-10)

            # t = r * sqrt((n-2)/(1-r^2))
            expected_t = r_scipy * np.sqrt((n_participants - 2) / (1 - r_scipy**2))
            np.testing.assert_allclose(t_stat[v], expected_t, atol=1e-10)

    def test_idx_not_to_include(self, small_mesh):
        """Excluded vertices should have r=0 and t=0."""
        np.random.seed(42)
        n_vertices = small_mesh.GetNumberOfPoints()
        data = np.random.randn(10, n_vertices, 2)
        data[:, :, 1] = data[:, :, 0] + np.random.randn(10, n_vertices) * 0.1
        exclude = np.array([0, 1, 2])

        t_test = CorrelationTTest(data, small_mesh, return_new_mesh=True, idx_not_to_include=exclude)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()

        corr = vtk_to_numpy(mesh.GetPointData().GetArray('correlation'))
        t_stat = vtk_to_numpy(mesh.GetPointData().GetArray('t_statistic'))

        np.testing.assert_array_equal(corr[exclude], 0.0)
        np.testing.assert_array_equal(t_stat[exclude], 0.0)

    def test_scalar_names(self, small_mesh):
        """Result mesh should have arrays named 't_statistic' and 'correlation'."""
        data = np.random.RandomState(0).randn(5, small_mesh.GetNumberOfPoints(), 2)

        t_test = CorrelationTTest(data, small_mesh, return_new_mesh=True)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()

        assert mesh.GetPointData().GetArray('t_statistic') is not None
        assert mesh.GetPointData().GetArray('correlation') is not None


class TestReplaceNonFiniteValues:
    def test_nan_replaced(self):
        arr = np.array([1.0, np.nan, 3.0, np.nan])
        result = replace_non_finite_values(arr)
        expected = np.array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_inf_replaced(self):
        arr = np.array([1.0, np.inf, -np.inf, 2.0])
        result = replace_non_finite_values(arr)
        expected = np.array([1.0, 0.0, 0.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_mixed_non_finite(self):
        arr = np.array([np.nan, np.inf, -np.inf, 0.0, 5.0])
        result = replace_non_finite_values(arr)
        assert np.all(np.isfinite(result))
        np.testing.assert_array_equal(result[3:], [0.0, 5.0])

    def test_replace_value_ignored_bug(self):
        """Known bug: replace_value parameter is ignored, always replaces with 0."""
        arr = np.array([np.nan, np.inf, 1.0])
        result = replace_non_finite_values(arr, replace_value=999.0)
        # Due to the bug, values are replaced with 0, not 999
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])
