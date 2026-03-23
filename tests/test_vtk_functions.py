import os
import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from pyKneeSPM.vtk_functions import (
    read_vtk,
    clean_polydata,
    get_icp_transform,
    apply_transform,
    transfer_mesh_scalars_get_weighted_average_n_closest,
)


class TestReadVtk:
    def test_read_example_file(self):
        """Read the bundled example file and verify it returns valid data."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'pyKneeSPM', 'data')
        filepath = os.path.join(data_dir, 'change_cart_thick.vtk')
        if not os.path.exists(filepath):
            pytest.skip("Example VTK file not found")

        mesh = read_vtk(filepath)
        assert isinstance(mesh, vtk.vtkPolyData)
        assert mesh.GetNumberOfPoints() > 0
        assert mesh.GetNumberOfCells() > 0


class TestCleanPolydata:
    def test_clean_returns_valid_mesh(self, flat_grid_mesh):
        """clean_polydata should return a valid mesh with same or fewer points."""
        cleaned = clean_polydata(flat_grid_mesh)
        assert isinstance(cleaned, vtk.vtkPolyData)
        assert cleaned.GetNumberOfPoints() <= flat_grid_mesh.GetNumberOfPoints()
        assert cleaned.GetNumberOfPoints() > 0

    def test_clean_already_clean_mesh(self, flat_grid_mesh):
        """Cleaning an already clean mesh should not change point count."""
        cleaned = clean_polydata(flat_grid_mesh)
        assert cleaned.GetNumberOfPoints() == flat_grid_mesh.GetNumberOfPoints()


class TestIcpTransformAndApply:
    def test_transform_moves_mesh(self, flat_grid_mesh):
        """Applying ICP transform between offset meshes should move the source."""
        # Create a translated copy
        translate = vtk.vtkTransform()
        translate.Translate(5.0, 5.0, 0.0)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(flat_grid_mesh)
        transform_filter.SetTransform(translate)
        transform_filter.Update()
        moved_mesh = vtk.vtkPolyData()
        moved_mesh.DeepCopy(transform_filter.GetOutput())

        icp = get_icp_transform(source=moved_mesh, target=flat_grid_mesh,
                                max_n_iter=100, n_landmarks=100, reg_mode='rigid')
        result = apply_transform(moved_mesh, icp)

        # The result should be close to the original
        orig_points = vtk_to_numpy(flat_grid_mesh.GetPoints().GetData())
        result_points = vtk_to_numpy(result.GetPoints().GetData())

        # Sort both by coordinates for comparison (ICP may reorder)
        orig_sorted = orig_points[np.lexsort(orig_points.T)]
        result_sorted = result_points[np.lexsort(result_points.T)]

        np.testing.assert_allclose(result_sorted, orig_sorted, atol=2.0)

    def test_identity_transform(self, flat_grid_mesh):
        """ICP of a mesh to itself should produce near-identity transform."""
        copy = vtk.vtkPolyData()
        copy.DeepCopy(flat_grid_mesh)

        icp = get_icp_transform(source=copy, target=flat_grid_mesh,
                                max_n_iter=100, n_landmarks=100, reg_mode='rigid')
        result = apply_transform(copy, icp)

        orig_points = vtk_to_numpy(flat_grid_mesh.GetPoints().GetData())
        result_points = vtk_to_numpy(result.GetPoints().GetData())

        np.testing.assert_allclose(result_points, orig_points, atol=0.5)


class TestTransferMeshScalars:
    def test_identical_meshes_transfer(self, small_mesh):
        """Transferring scalars between identical meshes should preserve values."""
        n_points = small_mesh.GetNumberOfPoints()
        values = np.arange(n_points, dtype=np.float64)
        scalars = numpy_to_vtk(values)
        scalars.SetName('test')
        small_mesh.GetPointData().SetScalars(scalars)

        target = vtk.vtkPolyData()
        target.DeepCopy(small_mesh)

        result = transfer_mesh_scalars_get_weighted_average_n_closest(target, small_mesh, n=3)
        np.testing.assert_allclose(result, values, atol=1e-6)

    def test_transfer_returns_correct_length(self, small_mesh):
        """Returned array should have length equal to new_mesh point count."""
        n_points = small_mesh.GetNumberOfPoints()
        values = np.ones(n_points)
        scalars = numpy_to_vtk(values)
        scalars.SetName('test')
        small_mesh.GetPointData().SetScalars(scalars)

        target = vtk.vtkPolyData()
        target.DeepCopy(small_mesh)

        result = transfer_mesh_scalars_get_weighted_average_n_closest(target, small_mesh, n=3)
        assert len(result) == n_points
