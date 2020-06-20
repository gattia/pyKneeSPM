import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np


class TestStatistics(object):
    def __init__(self,
                 data,
                 mesh,
                 return_new_mesh=True,
                 idx_not_to_include=None):
        self.data = data
        self.mesh = mesh
        self.return_new_mesh = return_new_mesh
        if self.return_new_mesh is True:
            self.test_statistics_mesh = vtk.vtkPolyData()
            self.test_statistics_mesh.DeepCopy(mesh)
        self.idx_not_to_include = idx_not_to_include
        self.test_statistics_array = None

    def get_statistics_array(self):
        return self.test_statistics_array

    def get_statistics_mesh(self):
        if self.return_new_mesh is True:
            return self.test_statistics_mesh
        elif self.return_new_mesh is False:
            # This is somewhat redundant, because self.mesh is just a view to the original mesh.
            # Therefore, the original mesh (Wherever it is) should still exist and have these values already.
            return self.mesh


class OneSampleZTest(TestStatistics):

    def compute_statistics_per_node(self):
        print('starting stats calc')
        mean_change = np.mean(self.data, axis=0)
        std_change = np.std(self.data, axis=0, ddof=1)
        se_change = std_change / np.sqrt(self.data.shape[0])
        z_score_change = mean_change / se_change
        if self.idx_not_to_include is not None:
            z_score_change[self.idx_not_to_include] = 0

        z_score_change = replace_non_finite_values(z_score_change, replace_value=0.)
        mean_change = replace_non_finite_values(mean_change, replace_value=0.)

        z_scalars = numpy_to_vtk(z_score_change)
        z_scalars.SetName('z_statistic')

        mean_change_scalars = numpy_to_vtk(mean_change)
        mean_change_scalars.SetName('mean_change')

        if self.return_new_mesh is True:
            self.test_statistics_mesh.GetPointData().AddArray(z_scalars)
            self.test_statistics_mesh.GetPointData().AddArray(mean_change_scalars)
            self.test_statistics_mesh.GetPointData().SetActiveScalars('z_statistic')
        elif self.return_new_mesh is False:
            self.mesh.GetPointData().AddArray(z_scalars)
            self.mesh.GetPointData().AddArray(mean_change_scalars)
            self.mesh.GetPointData().SetActiveScalars('z_statistic')


class CorrelationTTest(TestStatistics):
    def compute_statistics_per_node(self):
        residuals = self.data - self.data.mean(axis=0)
        numerator = np.sum(residuals[:, :, 0] * residuals[:, :, 1], axis=0)
        sd_1 = np.sqrt(np.sum(residuals[:, :, 0] ** 2, axis=0))
        sd_2 = np.sqrt(np.sum(residuals[:, :, 1] ** 2, axis=0))
        denominator = sd_1 * sd_2
        correlations = numerator / denominator
        t_statistic = correlations * np.sqrt((self.data.shape[0] - 2) / (1-correlations**2))

        if self.idx_not_to_include is not None:
            t_statistic[self.idx_not_to_include] = 0
            correlations[self.idx_not_to_include] = 0

        t_statistic = replace_non_finite_values(t_statistic, replace_value=0.)
        correlations = replace_non_finite_values(correlations, replace_value=0.)

        t_scalars = numpy_to_vtk(t_statistic)
        t_scalars.SetName('t_statistic')

        corr_scalars = numpy_to_vtk(correlations)
        corr_scalars.SetName('correlation')

        if self.return_new_mesh is True:
            self.test_statistics_mesh.GetPointData().AddArray(t_scalars)
            self.test_statistics_mesh.GetPointData().AddArray(corr_scalars)
            self.test_statistics_mesh.GetPointData().SetActiveScalars('t_statistic')
        elif self.return_new_mesh is False:
            self.mesh.GetPointData().AddArray(t_scalars)
            self.mesh.GetPointData().AddArray(corr_scalars)
            self.mesh.GetPointData().SetActiveScalars('t_statistic')

def replace_non_finite_values(array, replace_value=0.):
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = 0
    array[np.isneginf(array)] = 0
    return array
