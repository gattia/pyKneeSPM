import pytest
import numpy as np
import vtk
import os
import tempfile
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from scipy import stats

from pyKneeSPM.test_statistics import OneWayANOVA
from pyKneeSPM.monte_carlo import MonteCarloClusterANOVA
from pyKneeSPM.knee_spm import ANOVAComparison


# ── Helpers ──────────────────────────────────────────────────────────────


def make_simple_mesh(n_x=5, n_y=5):
    points = vtk.vtkPoints()
    for j in range(n_y):
        for i in range(n_x):
            points.InsertNextPoint(float(i), float(j), 0.0)
    cells = vtk.vtkCellArray()
    for j in range(n_y - 1):
        for i in range(n_x - 1):
            p0 = j * n_x + i
            p1 = p0 + 1
            p2 = (j + 1) * n_x + i
            p3 = p2 + 1
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, p0)
            tri.GetPointIds().SetId(1, p1)
            tri.GetPointIds().SetId(2, p2)
            cells.InsertNextCell(tri)
            tri2 = vtk.vtkTriangle()
            tri2.GetPointIds().SetId(0, p1)
            tri2.GetPointIds().SetId(1, p3)
            tri2.GetPointIds().SetId(2, p2)
            cells.InsertNextCell(tri2)
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh


def make_mesh_with_scalars(n_x, n_y, scalar_values, scalar_name='scalars'):
    mesh = make_simple_mesh(n_x, n_y)
    scalars = numpy_to_vtk(np.asarray(scalar_values, dtype=np.float64))
    scalars.SetName(scalar_name)
    mesh.GetPointData().SetScalars(scalars)
    return mesh


def write_vtk_mesh(mesh, filepath):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(mesh)
    writer.Write()


# ── OneWayANOVA ──────────────────────────────────────────────────────────


class TestOneWayANOVA:

    def test_known_group_difference(self, small_mesh):
        """Three groups with distinct means should produce large F-statistics."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data_a = np.random.normal(0.0, 0.5, (5, n_pts))
        data_b = np.random.normal(5.0, 0.5, (5, n_pts))
        data_c = np.random.normal(10.0, 0.5, (5, n_pts))
        data = np.vstack([data_a, data_b, data_c])
        labels = np.array([0]*5 + [1]*5 + [2]*5)

        anova = OneWayANOVA(data, small_mesh, return_new_mesh=True, group_labels=labels)
        anova.compute_statistics_per_node()
        mesh = anova.get_statistics_mesh()
        f_vals = vtk_to_numpy(mesh.GetPointData().GetArray('f_statistic'))
        # F should be very large with well-separated groups
        assert np.all(f_vals > 50.0)

    def test_no_group_difference(self, small_mesh):
        """Same distribution across groups should give small F-statistics."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(5.0, 1.0, (15, n_pts))
        labels = np.array([0]*5 + [1]*5 + [2]*5)

        anova = OneWayANOVA(data, small_mesh, return_new_mesh=True, group_labels=labels)
        anova.compute_statistics_per_node()
        mesh = anova.get_statistics_mesh()
        f_vals = vtk_to_numpy(mesh.GetPointData().GetArray('f_statistic'))
        # Under null, most F-values should be small
        assert np.median(f_vals) < 5.0

    def test_matches_scipy(self, small_mesh):
        """Verify against scipy.stats.f_oneway at each vertex."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data_a = np.random.normal(1.0, 2.0, (6, n_pts))
        data_b = np.random.normal(3.0, 2.0, (5, n_pts))
        data_c = np.random.normal(2.0, 2.0, (4, n_pts))
        data = np.vstack([data_a, data_b, data_c])
        labels = np.array([0]*6 + [1]*5 + [2]*4)

        anova = OneWayANOVA(data, small_mesh, return_new_mesh=True, group_labels=labels)
        anova.compute_statistics_per_node()
        mesh = anova.get_statistics_mesh()
        f_vals = vtk_to_numpy(mesh.GetPointData().GetArray('f_statistic'))

        for v in range(n_pts):
            scipy_f, _ = stats.f_oneway(data_a[:, v], data_b[:, v], data_c[:, v])
            np.testing.assert_allclose(f_vals[v], scipy_f, rtol=1e-10)

    def test_two_groups_matches_t_squared(self, small_mesh):
        """For 2 groups, F should equal t^2."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data_a = np.random.normal(3.0, 1.0, (5, n_pts))
        data_b = np.random.normal(1.0, 1.0, (5, n_pts))
        data = np.vstack([data_a, data_b])
        labels = np.array([0]*5 + [1]*5)

        anova = OneWayANOVA(data, small_mesh, return_new_mesh=True, group_labels=labels)
        anova.compute_statistics_per_node()
        f_vals = vtk_to_numpy(anova.get_statistics_mesh().GetPointData().GetArray('f_statistic'))

        for v in range(n_pts):
            scipy_t, _ = stats.ttest_ind(data_a[:, v], data_b[:, v], equal_var=True)
            np.testing.assert_allclose(f_vals[v], scipy_t**2, rtol=1e-10)

    def test_f_always_non_negative(self, small_mesh):
        """F-statistic should always be >= 0."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (12, n_pts))
        labels = np.array([0]*4 + [1]*4 + [2]*4)

        anova = OneWayANOVA(data, small_mesh, return_new_mesh=True, group_labels=labels)
        anova.compute_statistics_per_node()
        f_vals = vtk_to_numpy(anova.get_statistics_mesh().GetPointData().GetArray('f_statistic'))
        assert np.all(f_vals >= 0)

    def test_idx_not_to_include(self, small_mesh):
        """Excluded vertices should have F = 0."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (12, n_pts))
        data[:4] += 10
        labels = np.array([0]*4 + [1]*4 + [2]*4)
        exclude = np.array([0, 1, 2])

        anova = OneWayANOVA(data, small_mesh, return_new_mesh=True,
                            idx_not_to_include=exclude, group_labels=labels)
        anova.compute_statistics_per_node()
        f_vals = vtk_to_numpy(anova.get_statistics_mesh().GetPointData().GetArray('f_statistic'))
        assert np.all(f_vals[exclude] == 0)
        assert np.any(f_vals[3:] != 0)

    def test_non_finite_handling(self, small_mesh):
        """Zero within-group variance → division by zero → should be replaced with 0."""
        n_pts = small_mesh.GetNumberOfPoints()
        data = np.ones((12, n_pts))
        # All groups have identical values at vertex 0
        labels = np.array([0]*4 + [1]*4 + [2]*4)

        anova = OneWayANOVA(data, small_mesh, return_new_mesh=True, group_labels=labels)
        anova.compute_statistics_per_node()
        f_vals = vtk_to_numpy(anova.get_statistics_mesh().GetPointData().GetArray('f_statistic'))
        assert np.all(np.isfinite(f_vals))


# ── MonteCarloClusterANOVA ───────────────────────────────────────────────


class TestMonteCarloClusterANOVA:

    def test_smoke(self, small_mesh):
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (12, n_pts))
        labels = np.array([0]*4 + [1]*4 + [2]*4)

        mc = MonteCarloClusterANOVA(
            small_mesh, data,
            group_labels=labels,
            threshold=3.0,
            n_iterations=20,
        )
        mc.update()
        dist = mc.get_distribution_of_max_clustersizes()
        assert len(dist) == 20

    def test_threshold_non_negative(self, small_mesh):
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (12, n_pts))
        labels = np.array([0]*4 + [1]*4 + [2]*4)

        mc = MonteCarloClusterANOVA(
            small_mesh, data,
            group_labels=labels,
            threshold=3.0,
            n_iterations=20,
        )
        mc.update()
        thresh = mc.get_threshold_clustersize(threshold=0.05)
        assert thresh >= 0

    def test_uses_one_sided_threshold(self, small_mesh):
        """MonteCarloClusterANOVA should default to one_sided."""
        n_pts = small_mesh.GetNumberOfPoints()
        data = np.random.normal(0, 1, (12, n_pts))
        labels = np.array([0]*4 + [1]*4 + [2]*4)

        mc = MonteCarloClusterANOVA(
            small_mesh, data,
            group_labels=labels,
            threshold=3.0,
            n_iterations=10,
        )
        assert mc.threshold_type == 'one_sided'


# ── ANOVAComparison end-to-end ───────────────────────────────────────────


class TestANOVAComparisonE2E:

    def test_raw_array_3_groups(self, small_mesh):
        """End-to-end with 3 groups via raw numpy array."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)

        comp = ANOVAComparison(compute_min_cluster_size=False)
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}

        data_a = np.random.normal(0.0, 1.0, (5, n_pts))
        data_b = np.random.normal(5.0, 1.0, (5, n_pts))
        data_c = np.random.normal(10.0, 1.0, (5, n_pts))
        data = np.vstack([data_a, data_b, data_c])
        labels = np.array([0]*5 + [1]*5 + [2]*5)

        comp.add_change_data_directly(data, labels)
        comp.update()
        assert comp.map_name in comp.test_statistic_maps
        f_vals = vtk_to_numpy(
            comp.test_statistic_maps[comp.map_name].GetPointData().GetArray('f_statistic'))
        assert np.all(f_vals > 10.0)

    def test_raw_array_2_groups(self, small_mesh):
        """ANOVA should also work with exactly 2 groups."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)

        comp = ANOVAComparison(compute_min_cluster_size=False)
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}

        data = np.random.normal(0, 1, (10, n_pts))
        data[:5] += 5.0
        labels = np.array([0]*5 + [1]*5)

        comp.add_change_data_directly(data, labels)
        comp.update()
        assert comp.map_name in comp.test_statistic_maps

    def test_1_group_raises(self, small_mesh):
        comp = ANOVAComparison(compute_min_cluster_size=False)
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}
        data = np.ones((10, small_mesh.GetNumberOfPoints()))
        labels = np.array([0]*10)
        comp.add_change_data_directly(data, labels)
        with pytest.raises(ValueError, match='at least 2 groups'):
            comp.update()

    def test_prealigned_files(self):
        """End-to-end with pre-aligned VTK files, 3 groups."""
        n_pts = 25
        comp = ANOVAComparison(compute_min_cluster_size=False)

        np.random.seed(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(12):
                group = i % 3
                vals = np.random.normal(group * 5.0, 1.0, n_pts)
                mesh = make_mesh_with_scalars(5, 5, vals, 'thickness')
                path = os.path.join(tmpdir, f'mesh{i}.vtk')
                write_vtk_mesh(mesh, path)
                comp.add_prealigned_filepath(path, group_label=group, scalar_name='thickness')

            comp.update()
            assert comp.map_name in comp.test_statistic_maps

    def test_known_effect_with_mc(self, small_mesh):
        """Clear group difference should be detected with MC thresholding."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 0.1, (15, n_pts))
        # Large effect at first 10 vertices for group 0
        data[:5, :10] += 10.0
        labels = np.array([0]*5 + [1]*5 + [2]*5)

        comp = ANOVAComparison(
            map_threshold=3.0,
            compute_min_cluster_size=True,
            n_monte_carlo_iterations=50,
        )
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}
        comp.add_change_data_directly(data, labels)
        comp.update()

        assert len(comp.clustered_test_statistic_maps[comp.map_name]) > 0

    def test_no_effect_null(self, small_mesh):
        """No group difference → MC threshold should still be computed."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (15, n_pts))
        labels = np.array([0]*5 + [1]*5 + [2]*5)

        comp = ANOVAComparison(
            map_threshold=3.0,
            compute_min_cluster_size=True,
            n_monte_carlo_iterations=50,
        )
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}
        comp.add_change_data_directly(data, labels)
        comp.update()

        assert len(comp.threshold_cluster_size) > 0

    def test_one_sided_clustering(self, small_mesh):
        """ANOVAComparison should use one-sided clustering (F is always positive)."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (12, n_pts))
        data[:4] += 10.0
        labels = np.array([0]*4 + [1]*4 + [2]*4)

        comp = ANOVAComparison(map_threshold=3.0, compute_min_cluster_size=False)
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}
        comp.add_change_data_directly(data, labels)
        comp.update()

        # F-statistic map should exist and all values should be non-negative
        f_vals = vtk_to_numpy(
            comp.test_statistic_maps[comp.map_name].GetPointData().GetArray('f_statistic'))
        assert np.all(f_vals >= 0)
