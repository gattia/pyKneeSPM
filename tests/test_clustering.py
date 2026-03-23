import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from pyKneeSPM.clustering import Cluster
from tests.conftest import set_mesh_scalars


class TestCluster:
    def test_two_separated_clusters(self, flat_grid_mesh):
        """Two spatially separated high-value regions => two clusters found."""
        n_points = flat_grid_mesh.GetNumberOfPoints()
        scalars = np.zeros(n_points)

        # Region 1: top-left corner (points with x<3, y>6) => high positive
        # Region 2: bottom-right corner (points with x>6, y<3) => high positive
        for i in range(n_points):
            x, y, z = flat_grid_mesh.GetPoint(i)
            if x < 3 and y > 6:
                scalars[i] = 5.0
            elif x > 6 and y < 3:
                scalars[i] = 5.0

        set_mesh_scalars(flat_grid_mesh, scalars, name='z_statistic')
        flat_grid_mesh.GetPointData().SetActiveScalars('z_statistic')

        clust = Cluster(
            statistic_mesh=flat_grid_mesh,
            statistic_threshold=2.33,
            threshold_type='two_sided',
        )
        clusters = clust.get_clusters()
        areas = clust.get_areas()

        assert len(clusters) >= 2
        assert all(a > 0 for a in areas)

    def test_all_below_threshold_no_clusters(self, flat_grid_mesh):
        """All values below threshold => zero clusters."""
        n_points = flat_grid_mesh.GetNumberOfPoints()
        scalars = np.ones(n_points) * 1.0  # Below threshold of 2.33

        set_mesh_scalars(flat_grid_mesh, scalars, name='z_statistic')
        flat_grid_mesh.GetPointData().SetActiveScalars('z_statistic')

        clust = Cluster(
            statistic_mesh=flat_grid_mesh,
            statistic_threshold=2.33,
            threshold_type='two_sided',
        )
        clusters = clust.get_clusters()
        areas = clust.get_areas()

        assert len(clusters) == 0
        assert len(areas) == 0

    def test_all_above_threshold_one_cluster(self, flat_grid_mesh):
        """All values above threshold => one cluster covering the full mesh."""
        n_points = flat_grid_mesh.GetNumberOfPoints()
        scalars = np.ones(n_points) * 10.0

        set_mesh_scalars(flat_grid_mesh, scalars, name='z_statistic')
        flat_grid_mesh.GetPointData().SetActiveScalars('z_statistic')

        clust = Cluster(
            statistic_mesh=flat_grid_mesh,
            statistic_threshold=2.33,
            threshold_type='two_sided',
        )
        clusters = clust.get_clusters()
        areas = clust.get_areas()

        assert len(clusters) == 1
        # The total area should be close to the full mesh area (9x9 grid = 81)
        assert areas[0] > 0
        assert areas[0] == pytest.approx(81.0, rel=0.1)

    def test_areas_are_positive(self, flat_grid_mesh):
        """Cluster areas should always be positive."""
        n_points = flat_grid_mesh.GetNumberOfPoints()
        scalars = np.zeros(n_points)
        # Create one cluster
        for i in range(n_points):
            x, y, z = flat_grid_mesh.GetPoint(i)
            if x < 5 and y < 5:
                scalars[i] = 5.0

        set_mesh_scalars(flat_grid_mesh, scalars, name='z_statistic')
        flat_grid_mesh.GetPointData().SetActiveScalars('z_statistic')

        clust = Cluster(
            statistic_mesh=flat_grid_mesh,
            statistic_threshold=2.33,
            threshold_type='two_sided',
        )
        areas = clust.get_areas()
        for a in areas:
            assert a > 0

    def test_negative_clusters_two_sided(self, flat_grid_mesh):
        """Two-sided threshold should also find clusters of large negative values."""
        n_points = flat_grid_mesh.GetNumberOfPoints()
        scalars = np.zeros(n_points)
        for i in range(n_points):
            x, y, z = flat_grid_mesh.GetPoint(i)
            if x < 4 and y < 4:
                scalars[i] = -5.0  # Large negative

        set_mesh_scalars(flat_grid_mesh, scalars, name='z_statistic')
        flat_grid_mesh.GetPointData().SetActiveScalars('z_statistic')

        clust = Cluster(
            statistic_mesh=flat_grid_mesh,
            statistic_threshold=2.33,
            threshold_type='two_sided',
        )
        clusters = clust.get_clusters()
        assert len(clusters) >= 1

    def test_one_sided_ignores_negative(self, flat_grid_mesh):
        """One-sided threshold should not find negative clusters."""
        n_points = flat_grid_mesh.GetNumberOfPoints()
        scalars = np.ones(n_points) * (-5.0)  # All large negative

        set_mesh_scalars(flat_grid_mesh, scalars, name='z_statistic')
        flat_grid_mesh.GetPointData().SetActiveScalars('z_statistic')

        clust = Cluster(
            statistic_mesh=flat_grid_mesh,
            statistic_threshold=2.33,
            threshold_type='one_sided',
        )
        clusters = clust.get_clusters()
        assert len(clusters) == 0
