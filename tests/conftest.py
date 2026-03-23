import pytest
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


@pytest.fixture
def flat_grid_mesh():
    """Create a flat 10x10 triangulated grid with 100 points and known geometry.

    Points are on the z=0 plane, x and y in [0, 9].
    Each quad cell is split into 2 triangles, giving 162 triangles.
    Each triangle has area 0.5, so total mesh area = 81.
    """
    nx, ny = 10, 10
    points = vtk.vtkPoints()
    for j in range(ny):
        for i in range(nx):
            points.InsertNextPoint(float(i), float(j), 0.0)

    cells = vtk.vtkCellArray()
    for j in range(ny - 1):
        for i in range(nx - 1):
            p0 = j * nx + i
            p1 = p0 + 1
            p2 = (j + 1) * nx + i
            p3 = p2 + 1
            # Lower-left triangle
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, p0)
            tri.GetPointIds().SetId(1, p1)
            tri.GetPointIds().SetId(2, p2)
            cells.InsertNextCell(tri)
            # Upper-right triangle
            tri2 = vtk.vtkTriangle()
            tri2.GetPointIds().SetId(0, p1)
            tri2.GetPointIds().SetId(1, p3)
            tri2.GetPointIds().SetId(2, p2)
            cells.InsertNextCell(tri2)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh


@pytest.fixture
def small_mesh():
    """A smaller 5x5 flat grid (25 points, 32 triangles) for faster tests."""
    nx, ny = 5, 5
    points = vtk.vtkPoints()
    for j in range(ny):
        for i in range(nx):
            points.InsertNextPoint(float(i), float(j), 0.0)

    cells = vtk.vtkCellArray()
    for j in range(ny - 1):
        for i in range(nx - 1):
            p0 = j * nx + i
            p1 = p0 + 1
            p2 = (j + 1) * nx + i
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


def set_mesh_scalars(mesh, values, name="scalars"):
    """Helper: set point scalars on a mesh from a numpy array."""
    scalars = numpy_to_vtk(np.asarray(values, dtype=np.float64))
    scalars.SetName(name)
    mesh.GetPointData().SetScalars(scalars)
    return mesh
