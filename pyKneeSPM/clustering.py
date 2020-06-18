import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from .vtk_functions import clean_polydata
import numpy as np

class Cluster(object):
    def __init__(self,
                 statistic_mesh,
                 statistic_threshold=2.33,
                 threshold_type='two_sided',
                 clust_names='cluster',
                 clust_idx=0):
        self.statistic_mesh = statistic_mesh
        self.clust_names = clust_names
        self.clust_idx = clust_idx
        self.clusters = {}

        min_ = np.nanmin(vtk_to_numpy(statistic_mesh.GetPointData().GetScalars()))
        max_ = np.nanmax(vtk_to_numpy(statistic_mesh.GetPointData().GetScalars()))

        if threshold_type == 'two_sided':
            self.n_contours = 2
            self.statistic_thresholds = [min([min_-1, -statistic_threshold-1]),
                                         -statistic_threshold,
                                         statistic_threshold,
                                         max([statistic_threshold, max_+1])]
            self.contour_values = [0, 2]
        elif threshold_type == 'one_sided':
            self.n_contours = 1
            self.statistic_thresholds = [min([0, min_-1]),
                                        statistic_threshold,
                                        max(statistic_threshold+1, max_ + 1)]
            self.contour_values = [1]

        self.band_contour_filter()
        self.delete_cells_below_threshold()
        self.separate_individual_clusters()
        self.calc_cluster_sizes()

    def band_contour_filter(self):
        band_contour_filter = vtk.vtkBandedPolyDataContourFilter()
        band_contour_filter.SetInputDataObject(self.statistic_mesh)
        band_contour_filter.SetNumberOfContours(self.n_contours)
        for idx, value in enumerate(self.statistic_thresholds):
            band_contour_filter.SetValue(idx, value)
        band_contour_filter.Update()
        self.band_contour_mesh = vtk.vtkPolyData()

        self.band_contour_mesh.DeepCopy(band_contour_filter.GetOutput())
        self.band_contour_mesh.BuildLinks()

    def delete_cells_below_threshold(self):
        cell_scalars = vtk_to_numpy(self.band_contour_mesh.GetCellData().GetScalars())
        for cell_idx in range(self.band_contour_mesh.GetNumberOfCells()):
            if cell_scalars[cell_idx] in self.contour_values:
                pass
            else:
                self.band_contour_mesh.DeleteCell(cell_idx)
        self.band_contour_mesh.RemoveDeletedCells()

    def separate_individual_clusters(self):
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputDataObject(self.band_contour_mesh)
        connectivity.SetExtractionModeToSpecifiedRegions()
        while True:
            connectivity.AddSpecifiedRegion(self.clust_idx)
            connectivity.Update()

            clust = vtk.vtkPolyData()
            clust.DeepCopy(connectivity.GetOutput())

            cleaned_clust = clean_polydata(clust)

            if cleaned_clust.GetNumberOfCells() <= 0:
                break

            self.clusters['{}_{}'.format(self.clust_names, self.clust_idx)] = {}
            self.clusters['{}_{}'.format(self.clust_names, self.clust_idx)]['mesh'] = cleaned_clust
            connectivity.DeleteSpecifiedRegion(self.clust_idx)
            self.clust_idx += 1

    def calc_cluster_sizes(self):
        for cluster_idx, cluster_key in enumerate(self.clusters.keys()):
            cluster_mesh = self.clusters[cluster_key]['mesh']
            cell_size_filter = vtk.vtkCellSizeFilter()
            cell_size_filter.SetInputData(cluster_mesh)
            cell_size_filter.ComputeAreaOn()
            cell_size_filter.ComputeSumOn()
            cell_size_filter.Update()
            cell_area_mesh = cell_size_filter.GetOutput()
            total_cluster_area = vtk_to_numpy(cell_area_mesh.GetFieldData().GetAbstractArray('Area'))
            self.clusters[cluster_key]['area'] = total_cluster_area[0]

    def get_clusters(self):
        return self.clusters

    def get_areas(self):
        return [self.clusters[clust_key]['area'] for clust_key in self.clusters.keys()]

    def combine_clusters_to_one_map(self):
        pass