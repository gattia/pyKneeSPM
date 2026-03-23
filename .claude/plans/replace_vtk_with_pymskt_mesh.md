# Plan: Replace Raw VTK with pymskt.mesh.Mesh

## Goal

Replace all raw VTK I/O and utility code with `pymskt.mesh.Mesh`, which inherits from `pyvista.PolyData` and wraps `vtk.vtkPolyData`. This simplifies the codebase, eliminates `vtk_functions.py`, and aligns pyKneeSPM with pymskt for future integration.

## What pymskt.mesh.Mesh provides

`Mesh` inherits from `pv.PolyData`, so all pyvista and VTK methods are available. Key methods that replace current pyKneeSPM code:

- **I/O**: `Mesh(mesh=pv.read(path))`, `mesh.save_mesh(path)`
- **Scalars**: `mesh.get_scalar(name)`, `mesh.set_scalar(name, array)`, `mesh.point_data['name']`
- **Copy**: `mesh.copy()`
- **Clean**: `mesh.clean()` (inherited from pyvista)
- **ICP registration**: `mesh.rigidly_register(other_mesh, reg_mode='similarity')`
- **Non-rigid registration**: `mesh.non_rigidly_register(other_mesh, reg_method='focusr')`
- **Scalar transfer**: `mesh.copy_scalars_from_other_mesh_to_current()`
- **Connectivity**: `mesh.connectivity()` (inherited from pyvista)
- **Cell sizes**: `mesh.compute_cell_sizes()` (inherited from pyvista)

## Changes by file

### 1. `vtk_functions.py` — delete entirely

All functions are replaced by Mesh methods:
- `read_vtk()` → `Mesh(mesh=pv.read(path))`
- `clean_polydata()` → `mesh.clean()`
- `get_icp_transform()` + `apply_transform()` → `mesh.rigidly_register()`
- `transfer_mesh_scalars_get_weighted_average_n_closest()` → `mesh.copy_scalars_from_other_mesh_to_current()`

### 2. `test_statistics.py` — simplify scalar handling

Replace:
```python
vtk.vtkPolyData(); pd.DeepCopy(mesh)    →  mesh.copy()
numpy_to_vtk(arr); SetScalars(vtk_arr)  →  mesh.set_scalar('z_statistic', arr)
vtk_to_numpy(mesh.GetPointData()...)     →  mesh.get_scalar('name')
```

The `TestStatistics` base class constructor currently does a `DeepCopy` — replace with `mesh.copy()`. The `return_new_mesh` flag stays the same, just uses cleaner methods.

### 3. `clustering.py` — partial replacement

- Replace `clean_polydata(clust)` import with `clust.clean()`.
- Replace `vtk_to_numpy(mesh.GetPointData().GetScalars())` with `mesh.active_scalars` or `mesh.get_scalar()`.
- `vtkBandedPolyDataContourFilter` has no pyvista wrapper — keep raw VTK for this filter, but convert input/output to Mesh objects. Since Mesh inherits from vtkPolyData, it can be passed directly to VTK filters.
- `vtkCellSizeFilter` → `mesh.compute_cell_sizes()` from pyvista.
- `vtkPolyDataConnectivityFilter` → `mesh.connectivity()` from pyvista (check if extraction mode is flexible enough, otherwise keep raw VTK).

### 4. `knee_spm.py` — simplify compile_data

The `compile_data()` methods currently do ~30 lines of manual pyfocusr setup. With Mesh:
```python
target = Mesh(mesh=pv.read(filepath))
target.rigidly_register(self.reference_mesh, reg_mode='similarity')
target.non_rigidly_register(self.reference_mesh, reg_method='focusr')
# scalar transfer is handled by the registration
```

This also removes the lazy `import pyfocusr` — pymskt handles that internally.

### 5. `monte_carlo.py` — minor scalar access changes

Replace `vtk_to_numpy(mesh.GetPointData().GetArray('name'))` with `mesh.get_scalar('name')`.

### 6. `__init__.py` — remove vtk_functions import

Remove `from . import vtk_functions`.

### 7. `setup.py` / `requirements.txt` — update dependencies

- Add `pymskt` (which brings in `pyvista` transitively).
- Remove direct `vtk` dependency (pulled in by pyvista/pymskt).
- Remove `pyfocusr` entirely (pymskt handles it).

## Migration notes

- Since `Mesh` inherits from `pv.PolyData` which inherits from `vtk.vtkPolyData`, any code that still needs raw VTK access (like `vtkBandedPolyDataContourFilter`) can use the Mesh object directly — no wrapping/unwrapping needed.
- The `reference_mesh` dict stored throughout `knee_spm.py` currently holds `{'mesh': vtkPolyData, 'filename': str, 'id': ...}`. The `'mesh'` value becomes a `Mesh` object instead. Everything downstream is compatible since Mesh is-a vtkPolyData.
- This should be done **after** the test suite is in place so we can verify nothing breaks.
