import pytest
import numpy as np
import vtk
import os
import tempfile
from math import comb, factorial
from functools import reduce
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from scipy import stats

from pyKneeSPM.test_statistics import IndependentSamplesTTest
from pyKneeSPM.permutation_test import PermutationGroupLabels
from pyKneeSPM.monte_carlo import MonteCarloClusterGroupTest, MonteCarloClusterTwoSampleTest
from pyKneeSPM.knee_spm import GroupComparison, TwoSampleComparison
from pyKneeSPM.vtk_functions import read_vtk


# ── Helpers ──────────────────────────────────────────────────────────────


def make_simple_mesh(n_x=5, n_y=5):
    """Create a flat triangulated grid with n_x * n_y vertices."""
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
    """Create a mesh and attach scalar values to it."""
    mesh = make_simple_mesh(n_x, n_y)
    scalars = numpy_to_vtk(np.asarray(scalar_values, dtype=np.float64))
    scalars.SetName(scalar_name)
    mesh.GetPointData().SetScalars(scalars)
    return mesh


def write_vtk_mesh(mesh, filepath):
    """Write a VTK polydata mesh to a .vtk file."""
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(mesh)
    writer.Write()


# ── IndependentSamplesTTest ──────────────────────────────────────────────


class TestIndependentSamplesTTest:

    def test_known_large_difference(self, small_mesh):
        """Groups with large mean difference should produce large t-statistics."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data_a = np.random.normal(10.0, 1.0, (5, n_pts))
        data_b = np.random.normal(0.0, 1.0, (5, n_pts))
        data = np.vstack([data_a, data_b])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        t_test = IndependentSamplesTTest(data, small_mesh, return_new_mesh=True, group_labels=labels)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()
        t_vals = vtk_to_numpy(mesh.GetPointData().GetArray('t_statistic'))
        # All t-values should be large positive (group 0 > group 1)
        assert np.all(t_vals > 5.0)

    def test_zero_difference(self, small_mesh):
        """Same mean in both groups should give t-stats near zero."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(5.0, 1.0, (10, n_pts))
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        t_test = IndependentSamplesTTest(data, small_mesh, return_new_mesh=True, group_labels=labels)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()
        t_vals = vtk_to_numpy(mesh.GetPointData().GetArray('t_statistic'))
        # t-values should be near zero (within sampling noise)
        assert np.all(np.abs(t_vals) < 4.0)

    def test_matches_scipy(self, small_mesh):
        """Verify against scipy.stats.ttest_ind at each vertex."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data_a = np.random.normal(3.0, 2.0, (6, n_pts))
        data_b = np.random.normal(1.0, 2.0, (4, n_pts))
        data = np.vstack([data_a, data_b])
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

        t_test = IndependentSamplesTTest(data, small_mesh, return_new_mesh=True, group_labels=labels)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()
        t_vals = vtk_to_numpy(mesh.GetPointData().GetArray('t_statistic'))

        # Compare with scipy
        for v in range(n_pts):
            scipy_t, _ = stats.ttest_ind(data_a[:, v], data_b[:, v], equal_var=True)
            np.testing.assert_allclose(t_vals[v], scipy_t, rtol=1e-10)

    def test_idx_not_to_include(self, small_mesh):
        """Excluded vertices should have t-statistic = 0."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (10, n_pts))
        data[:5] += 10  # group 0 has higher values
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        exclude = np.array([0, 1, 2])

        t_test = IndependentSamplesTTest(
            data, small_mesh, return_new_mesh=True,
            idx_not_to_include=exclude, group_labels=labels)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()
        t_vals = vtk_to_numpy(mesh.GetPointData().GetArray('t_statistic'))
        assert np.all(t_vals[exclude] == 0)
        # Non-excluded vertices should still have values
        assert np.any(t_vals[3:] != 0)

    def test_non_finite_handling(self, small_mesh):
        """Zero variance in one group would cause inf — should be replaced with 0."""
        n_pts = small_mesh.GetNumberOfPoints()
        data = np.ones((10, n_pts))
        data[5:] = 2.0
        # Make one vertex have zero variance in both groups → 0/0
        data[:, 0] = 1.0
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        t_test = IndependentSamplesTTest(data, small_mesh, return_new_mesh=True, group_labels=labels)
        t_test.compute_statistics_per_node()
        mesh = t_test.get_statistics_mesh()
        t_vals = vtk_to_numpy(mesh.GetPointData().GetArray('t_statistic'))
        assert np.all(np.isfinite(t_vals))


# ── PermutationGroupLabels ───────────────────────────────────────────────


class TestPermutationGroupLabels:

    def test_output_shape(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # C(10,5) = 252 > 100
        perm = PermutationGroupLabels(group_labels=labels, n_permutations=100)
        perm.update()
        shuffled = perm.get_shuffled_labels()
        assert shuffled.shape == (100, 10)

    def test_group_sizes_preserved(self):
        """Every permutation row must have the same count of each label."""
        labels = np.array([0, 0, 0, 1, 1])
        perm = PermutationGroupLabels(group_labels=labels, n_permutations=50)
        perm.update()
        shuffled = perm.get_shuffled_labels()
        for row in shuffled:
            assert np.sum(row == 0) == 3
            assert np.sum(row == 1) == 2

    def test_exhaustive_small_case(self):
        """For 4 participants, 2 per group → C(4,2)=6 unique permutations."""
        labels = np.array([0, 0, 1, 1])
        perm = PermutationGroupLabels(group_labels=labels, n_permutations=1000)
        perm.update()
        shuffled = perm.get_shuffled_labels()
        assert shuffled.shape[0] == 6
        # All rows should be unique
        unique_rows = set(tuple(row) for row in shuffled)
        assert len(unique_rows) == 6

    def test_unique_permutation_count_2_groups(self):
        labels = np.array([0, 0, 0, 1, 1])
        perm = PermutationGroupLabels(group_labels=labels, n_permutations=10)
        assert perm.n_possible_unique_permutations == comb(5, 3)

    def test_unique_permutation_count_3_groups(self):
        labels = np.array([0, 0, 1, 1, 2])
        perm = PermutationGroupLabels(group_labels=labels, n_permutations=10)
        expected = factorial(5) // (factorial(2) * factorial(2) * factorial(1))
        assert perm.n_possible_unique_permutations == expected


# ── MonteCarloClusterGroupTest / MonteCarloClusterTwoSampleTest ──────────


class TestMonteCarloClusterTwoSampleTest:

    def test_smoke(self, small_mesh):
        """MC simulation completes and produces a cluster size distribution."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (10, n_pts))
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        mc = MonteCarloClusterTwoSampleTest(
            small_mesh, data,
            group_labels=labels,
            threshold=2.0,
            n_iterations=20,
        )
        mc.update()
        dist = mc.get_distribution_of_max_clustersizes()
        assert len(dist) == 20

    def test_threshold_non_negative(self, small_mesh):
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (10, n_pts))
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        mc = MonteCarloClusterTwoSampleTest(
            small_mesh, data,
            group_labels=labels,
            threshold=2.0,
            n_iterations=20,
        )
        mc.update()
        thresh = mc.get_threshold_clustersize(threshold=0.05)
        assert thresh >= 0

    def test_uses_shuffled_labels(self, small_mesh):
        """Verify shuffled_labels is populated and rand_permutations is None."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (10, n_pts))
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        mc = MonteCarloClusterTwoSampleTest(
            small_mesh, data,
            group_labels=labels,
            threshold=2.0,
            n_iterations=20,
        )
        mc.update()
        assert mc.shuffled_labels is not None
        assert mc.rand_permutations is None


# ── GroupComparison / TwoSampleComparison ────────────────────────────────


class TestGroupComparisonAddPrealigned:

    def test_first_mesh_sets_reference(self, small_mesh):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)
        n_pts = small_mesh.GetNumberOfPoints()

        with tempfile.TemporaryDirectory() as tmpdir:
            mesh = make_mesh_with_scalars(5, 5, np.ones(n_pts), 'thickness')
            path = os.path.join(tmpdir, 'mesh0.vtk')
            write_vtk_mesh(mesh, path)
            comp.add_prealigned_filepath(path, group_label=0, scalar_name='thickness')
            assert 'mesh' in comp.reference_mesh
            assert comp.reference_mesh['mesh'].GetNumberOfPoints() == n_pts

    def test_matching_coords_accepted(self, small_mesh):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)
        n_pts = small_mesh.GetNumberOfPoints()

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                mesh = make_mesh_with_scalars(5, 5, np.full(n_pts, float(i)), 'val')
                path = os.path.join(tmpdir, f'mesh{i}.vtk')
                write_vtk_mesh(mesh, path)
                comp.add_prealigned_filepath(path, group_label=i % 2, scalar_name='val')
            assert len(comp._prealigned_scalars) == 3

    def test_different_coords_raises(self):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # First mesh: normal grid
            mesh1 = make_mesh_with_scalars(5, 5, np.ones(25), 'val')
            path1 = os.path.join(tmpdir, 'mesh1.vtk')
            write_vtk_mesh(mesh1, path1)
            comp.add_prealigned_filepath(path1, group_label=0, scalar_name='val')

            # Second mesh: shifted coordinates
            mesh2 = make_simple_mesh(5, 5)
            pts = vtk_to_numpy(mesh2.GetPoints().GetData())
            pts += 100.0  # shift all coordinates
            new_points = vtk.vtkPoints()
            for p in pts:
                new_points.InsertNextPoint(p[0], p[1], p[2])
            mesh2.SetPoints(new_points)
            scalars = numpy_to_vtk(np.ones(25, dtype=np.float64))
            scalars.SetName('val')
            mesh2.GetPointData().SetScalars(scalars)
            path2 = os.path.join(tmpdir, 'mesh2.vtk')
            write_vtk_mesh(mesh2, path2)

            with pytest.raises(ValueError, match='coordinates do not match'):
                comp.add_prealigned_filepath(path2, group_label=1, scalar_name='val')

    def test_different_vertex_count_raises(self):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            mesh1 = make_mesh_with_scalars(5, 5, np.ones(25), 'val')
            path1 = os.path.join(tmpdir, 'mesh1.vtk')
            write_vtk_mesh(mesh1, path1)
            comp.add_prealigned_filepath(path1, group_label=0, scalar_name='val')

            mesh2 = make_mesh_with_scalars(3, 3, np.ones(9), 'val')
            path2 = os.path.join(tmpdir, 'mesh2.vtk')
            write_vtk_mesh(mesh2, path2)

            with pytest.raises(ValueError, match='vertices'):
                comp.add_prealigned_filepath(path2, group_label=1, scalar_name='val')

    def test_scalar_extraction_by_name(self):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            vals = np.arange(25, dtype=np.float64)
            mesh = make_mesh_with_scalars(5, 5, vals, 'thickness')
            path = os.path.join(tmpdir, 'mesh.vtk')
            write_vtk_mesh(mesh, path)
            comp.add_prealigned_filepath(path, group_label=0, scalar_name='thickness')
            np.testing.assert_array_almost_equal(comp._prealigned_scalars[0], vals)

    def test_scalar_extraction_active(self):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            vals = np.arange(25, dtype=np.float64)
            mesh = make_mesh_with_scalars(5, 5, vals, 'scalars')
            path = os.path.join(tmpdir, 'mesh.vtk')
            write_vtk_mesh(mesh, path)
            comp.add_prealigned_filepath(path, group_label=0, scalar_name=None)
            np.testing.assert_array_almost_equal(comp._prealigned_scalars[0], vals)

    def test_group_labels_stored(self):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(4):
                mesh = make_mesh_with_scalars(5, 5, np.ones(25) * i, 'val')
                path = os.path.join(tmpdir, f'mesh{i}.vtk')
                write_vtk_mesh(mesh, path)
                comp.add_prealigned_filepath(path, group_label=i % 2, scalar_name='val')
            assert comp.group_labels_list == [0, 1, 0, 1]


class TestGroupComparisonAddDataDirectly:

    def test_mismatched_labels_raises(self, small_mesh):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)
        data = np.ones((10, 25))
        labels = np.array([0, 0, 1])  # wrong length
        with pytest.raises(ValueError, match='group labels'):
            comp.add_change_data_directly(data, labels)

    def test_mismatched_vertices_raises(self, small_mesh):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)
        comp.reference_mesh = {'mesh': small_mesh}
        data = np.ones((10, 50))  # 50 vertices != 25
        labels = np.array([0] * 5 + [1] * 5)
        with pytest.raises(ValueError, match='vertices'):
            comp.add_change_data_directly(data, labels)

    def test_valid_data_stored(self, small_mesh):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)
        data = np.ones((10, 25))
        labels = np.array([0] * 5 + [1] * 5)
        comp.add_change_data_directly(data, labels)
        np.testing.assert_array_equal(comp.change_values, data)
        np.testing.assert_array_equal(comp.group_labels, labels)


class TestFinalizePrealignedData:

    def test_stacks_correctly(self):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(4):
                mesh = make_mesh_with_scalars(5, 5, np.ones(25) * i, 'val')
                path = os.path.join(tmpdir, f'mesh{i}.vtk')
                write_vtk_mesh(mesh, path)
                comp.add_prealigned_filepath(path, group_label=i % 2, scalar_name='val')

            comp._finalize_prealigned_data()
            assert comp.change_values.shape == (4, 25)
            assert len(comp.group_labels) == 4

    def test_noop_when_empty(self):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)
        comp._finalize_prealigned_data()
        assert comp.change_values is None


class TestValidateGroupLabels:

    def test_raises_when_none(self):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)
        comp.change_values = np.ones((10, 25))
        with pytest.raises(ValueError, match='group_labels must be set'):
            comp._validate_group_labels()

    def test_raises_when_count_mismatch(self):
        comp = GroupComparison(map_name='t_statistic', compute_min_cluster_size=False)
        comp.change_values = np.ones((10, 25))
        comp.group_labels = np.array([0, 1])
        with pytest.raises(ValueError, match='must match'):
            comp._validate_group_labels()


# ── TwoSampleComparison end-to-end ──────────────────────────────────────


class TestTwoSampleComparisonE2E:

    def test_mode_a_prealigned_files(self):
        """End-to-end with pre-aligned VTK files."""
        n_pts = 25
        comp = TwoSampleComparison(compute_min_cluster_size=False)

        np.random.seed(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                group = 0 if i < 5 else 1
                vals = np.random.normal(10.0 if group == 0 else 0.0, 1.0, n_pts)
                mesh = make_mesh_with_scalars(5, 5, vals, 'thickness')
                path = os.path.join(tmpdir, f'mesh{i}.vtk')
                write_vtk_mesh(mesh, path)
                comp.add_prealigned_filepath(path, group_label=group, scalar_name='thickness')

            comp.update()
            assert comp.map_name in comp.test_statistic_maps
            t_mesh = comp.test_statistic_maps[comp.map_name]
            t_vals = vtk_to_numpy(t_mesh.GetPointData().GetArray('t_statistic'))
            # Strong group difference → large t-values
            assert np.all(np.abs(t_vals) > 3.0)

    def test_mode_b_raw_array(self, small_mesh):
        """End-to-end with raw numpy array."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)

        comp = TwoSampleComparison(compute_min_cluster_size=False)
        comp.add_reference_change_mesh.__func__  # just checking it exists

        # Manually set reference mesh (bypassing file-based method)
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}

        data_a = np.random.normal(5.0, 1.0, (5, n_pts))
        data_b = np.random.normal(0.0, 1.0, (5, n_pts))
        data = np.vstack([data_a, data_b])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        comp.add_change_data_directly(data, labels)
        comp.update()
        assert comp.map_name in comp.test_statistic_maps

    def test_group_count_validation_1_group(self, small_mesh):
        """Passing 1 group should raise."""
        comp = TwoSampleComparison(compute_min_cluster_size=False)
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}
        data = np.ones((10, small_mesh.GetNumberOfPoints()))
        labels = np.array([0] * 10)
        comp.add_change_data_directly(data, labels)
        with pytest.raises(ValueError, match='exactly 2 groups'):
            comp.update()

    def test_group_count_validation_3_groups(self, small_mesh):
        """Passing 3 groups should raise."""
        comp = TwoSampleComparison(compute_min_cluster_size=False)
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}
        data = np.ones((9, small_mesh.GetNumberOfPoints()))
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        comp.add_change_data_directly(data, labels)
        with pytest.raises(ValueError, match='exactly 2 groups'):
            comp.update()

    def test_known_effect_detected(self, small_mesh):
        """Clear group difference at specific vertices should produce significant clusters."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 0.1, (20, n_pts))
        # Add large effect at first 10 vertices for group 0
        data[:10, :10] += 5.0
        labels = np.array([0] * 10 + [1] * 10)

        comp = TwoSampleComparison(
            map_threshold=2.0,
            compute_min_cluster_size=True,
            n_monte_carlo_iterations=50,
        )
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}
        comp.add_change_data_directly(data, labels)
        comp.update()

        # Check that clusters were found
        assert len(comp.clustered_test_statistic_maps[comp.map_name]) > 0

    def test_no_effect_null(self, small_mesh):
        """No group difference → no significant clusters (ideally)."""
        n_pts = small_mesh.GetNumberOfPoints()
        np.random.seed(42)
        data = np.random.normal(0, 1, (20, n_pts))
        labels = np.array([0] * 10 + [1] * 10)

        comp = TwoSampleComparison(
            map_threshold=2.33,
            compute_min_cluster_size=True,
            n_monte_carlo_iterations=50,
        )
        comp.reference_mesh = {'mesh': small_mesh, 'filename': None, 'id': None}
        comp.add_change_data_directly(data, labels)
        comp.update()

        # Under null, threshold cluster size should be defined
        assert len(comp.threshold_cluster_size) > 0
