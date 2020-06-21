import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np

def clean_polydata(polydata):
    clean_poly = vtk.vtkCleanPolyData()
    clean_poly.SetInputData(polydata)
    clean_poly.Update()

    cleaned_polydata = vtk.vtkPolyData()
    cleaned_polydata.DeepCopy(clean_poly.GetOutput())

    return cleaned_polydata


def read_vtk(filepath):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    return reader.GetOutput()


def get_icp_transform(source, target, max_n_iter=1000, n_landmarks=1000, reg_mode='similarity'):
    """
    transform = ('rigid': true rigid, translation only; similarity': rigid + equal scale)
    """
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    if reg_mode == 'rigid':
        icp.GetLandmarkTransform().SetModeToRigidBody()
    elif reg_mode == 'similarity':
        icp.GetLandmarkTransform().SetModeToSimilarity()
    icp.SetMaximumNumberOfIterations(max_n_iter)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    icp.SetMaximumNumberOfLandmarks(n_landmarks)
    return icp


def apply_transform(source, transform):
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(source)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()


def transfer_mesh_scalars_get_weighted_average_n_closest(new_mesh, old_mesh, n=3):
    kDTree = vtk.vtkKdTreePointLocator()
    kDTree.SetDataSet(old_mesh)
    kDTree.BuildLocator()

    new_scalars = np.zeros(new_mesh.GetNumberOfPoints())
    scalars_old_mesh = np.copy(vtk_to_numpy(old_mesh.GetPointData().GetScalars()))
    for new_mesh_pt_idx in range(new_mesh.GetNumberOfPoints()):
        point = new_mesh.GetPoint(new_mesh_pt_idx)
        closest_ids = vtk.vtkIdList()
        kDTree.FindClosestNPoints(n, point, closest_ids)

        list_scalars = []
        distance_weighting = []
        for closest_pts_idx in range(closest_ids.GetNumberOfIds()):
            pt_idx = closest_ids.GetId(closest_pts_idx)
            _point = old_mesh.GetPoint(pt_idx)
            list_scalars.append(scalars_old_mesh[pt_idx])
            distance_weighting.append(1 / np.sqrt(np.sum(np.square(np.asarray(point) - np.asarray(_point)))))

        total_distance = np.sum(distance_weighting)
        normalized_value = np.sum(np.asarray(list_scalars) * np.asarray(distance_weighting)) / total_distance
        new_scalars[new_mesh_pt_idx] = normalized_value
    return new_scalars


def transfer_clusters_to_ref_mesh(ref_mesh, cluster_mesh):
    kDTree = vtk.vtkKdTreePointLocator()
    kDTree.SetDataSet(ref_mesh)
    kDTree.BuildLocator()

    # Create new mesh to store results.
    new_mesh = vtk.vtkPolyData()
    new_mesh.DeepCopy(ref_mesh)
    new_mesh = delete_arrays_mesh(new_mesh)

    # Create dicts to store existing scalars & pre-allocate empty arrays for scalars on new mesh.
    new_mesh_arrays = {}
    cluster_mesh_arrays = {}
    n_arrays = cluster_mesh.GetPointData().GetNumberOfArrays()
    # Store array names
    array_names = []

    for array_idx in range(n_arrays):
        array_name = cluster_mesh.GetPointData().GetArrayName(array_idx)
        new_mesh_arrays[array_name] = np.zeros(ref_mesh.GetNumberOfPoints())
        cluster_mesh_arrays[array_name] = vtk_to_numpy(cluster_mesh.GetPointData().GetArray(array_name))
        array_names.append(array_name)


    # Iterate over every point on the cluster mesh:
    for clust_mesh_pt_idx in range(cluster_mesh.GetNumberOfPoints()):
        point = cluster_mesh.GetPoint(clust_mesh_pt_idx)
        # Find the closest point on the ref_mesh
        closest_pt = kDTree.FindClosestPoint(point)
        # Iterate over every array in cluster mesh.
        for array_name in array_names:
            # Get the value of the array at the clust point.
            value = cluster_mesh_arrays[array_name][clust_mesh_pt_idx]
            # Assign value to new_mesh array of same name
            new_mesh_arrays[array_name][closest_pt] = value

    # Iterate over each array name
    for array_name in array_names:
        # create vtk array
        scalars = numpy_to_vtk(new_mesh_arrays[array_name])
        # rename based on the array_name
        scalars.SetName(array_name)
        # assign array to new_mesh.
        new_mesh.GetPointData().AddArray(scalars)

    return new_mesh


def delete_arrays_mesh(mesh):
    n_arrays = mesh.GetPointData().GetNumberOfArrays()
    array_names = []
    for array_idx in range(n_arrays):
        array_names.append(mesh.GetPointData().GetArrayName(array_idx))
    for array_name in array_names:
        mesh.GetPointData().RemoveArray(array_name)
    return mesh


