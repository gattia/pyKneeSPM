import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from .vtk_functions import (read_vtk,
                            apply_transform,
                            get_icp_transform,
                            transfer_mesh_scalars_get_weighted_average_n_closest,
                            transfer_clusters_to_ref_mesh)
from . import test_statistics
from .clustering import Cluster
from .monte_carlo import *
import pyfocusr


class SPM(object):
    def __init__(self,
                 ):

        self.dict_meshes = {}
        self.participant_idx = 0
        self.reference_mesh = {}
        self.test_statistic_maps = {}
        self.clustered_test_statistic_maps = {}
        self.combined_clustered_meshes = {}

    def get_test_statistic_maps(self):
        return self.test_statistic_maps

    def get_clustered_statistic_maps(self):
        return self.clustered_test_statistic_maps


class SingleStatisticSPM(SPM):
    def __init__(self,
                 *args,
                 map_name='test_map',
                 map_threshold=2.33,
                 find_ref_mesh=False,
                 find_ref_mesh_mode='similarity',
                 compute_min_cluster_size=True,
                 mc_cluster_method='permutation',
                 mc_cluster_extent_significance=0.05,
                 mc_point_significance=0.05,
                 n_monte_carlo_iterations=10000,
                 idx_no_data=None,
                 percent_participants_with_data_to_include_vertex=0.5,
                 registration_max_iterations=1000,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.clustered_test_statistic_maps[map_name] = {}
        self.map_name = map_name
        self.map_threshold = map_threshold if type(map_threshold) in (list, tuple) else [map_threshold,]
        self.find_ref_mesh = find_ref_mesh
        self.find_ref_mesh_mode = find_ref_mesh_mode
        self.reference_mesh = {}
        self.change_values = None
        self.n_participants = None
        self.n_points = None

        # Cluster size
        self.compute_min_cluster_size = compute_min_cluster_size
        self.mc_cluster_method = mc_cluster_method
        self.mc_cluster_extent_significance = mc_cluster_extent_significance
        self.n_monte_carlo_iterations = n_monte_carlo_iterations

        # Individual voxel statistic
        self.mc_point_significance = mc_point_significance
        self.threshold_test_statistic = {}

        self.idx_no_data = idx_no_data
        self.percent_participants_with_data_to_include_vertex = percent_participants_with_data_to_include_vertex
        self.added_change_values_directly = False

        # Registration parameters
        self.registration_max_iterations = registration_max_iterations

        self.threshold_cluster_distribution = {}
        self.threshold_cluster_size = {}
        self.threshold_test_statistic = {}
        self.sig_clusters = {}
        self.combined_sig_clustered_meshes = {}
        self.sig_clusters_meshes = {}

    def calc_significant_clusters(self):
        # This first line is specific to the onesample test... need to extend for multi statistic tests
        if self.threshold_cluster_size is None:
            # The following function doesnt exist....
            # self.compute_threshold_clustersize()
            self.compute_mc_thresholds()

        for stat_key in self.clustered_test_statistic_maps.keys():
            self.sig_clusters[stat_key] = {}
            for thresh_key in self.clustered_test_statistic_maps[stat_key].keys():
                self.sig_clusters[stat_key][thresh_key] = {}
                for clust_key in self.clustered_test_statistic_maps[stat_key][thresh_key].keys():
                    if self.clustered_test_statistic_maps[stat_key][thresh_key][clust_key]['area'] >= self.threshold_cluster_size[thresh_key]:
                        self.sig_clusters[stat_key][thresh_key][clust_key] = self.clustered_test_statistic_maps[stat_key][thresh_key][clust_key]

    def get_cluster_distributions(self):
        return self.threshold_cluster_distribution

    def get_significant_clusters(self):
        self.calc_significant_clusters()
        return self.sig_clusters

    def get_n_significant_clusters(self):
        # This first line is specific to the onesample test... need to extend for multi statistic tests
        self.calc_significant_clusters()

        n_sig_clusters = {}
        for stat_key in self.clustered_test_statistic_maps.keys():
            n_sig_clusters[stat_key] = {}
            for thresh_key in self.clustered_test_statistic_maps[stat_key].keys():
                n_sig_clusters[stat_key][thresh_key] = len(self.sig_clusters[stat_key][thresh_key])
        return n_sig_clusters

    def get_threshold_cluster_size(self):
        return self.threshold_cluster_size

    def get_threshold_test_statistic(self):
        return self.threshold_test_statistic

    def get_n_significant_individual_points(self):
        scalars = vtk_to_numpy(self.test_statistic_maps[self.map_name].GetPointData().GetAray(self.map_name))
        n_sig_points = len(np.where(scalars > self.threshold_test_statistic[self.map_name])[0])
        return n_sig_points

    def add_map_threshold(self, threshold):
        self.map_threshold.append(threshold)

    def add_reference_change_mesh(self,
                                  filepath,
                                  id=None):
        self.reference_mesh = {'filename': filepath,
                               'mesh': read_vtk(filepath),
                               'id': id
                               }

    def add_reference_pre_mesh(self,
                               filepath,
                               id=None):
        self.reference_mesh = {'filename': filepath,
                               'mesh': read_vtk(filepath),
                               'id': id
                               }

    def add_change_filepath(self,
                            change_filepath,
                            participant_identifier=None,
                            reference_mesh=False):
        id = participant_identifier or self.participant_idx
        if id in self.dict_meshes:
            self.dict_meshes[id]['change'] = {'filename': change_filepath}
        else:
            self.dict_meshes[id] = {'change': {'filename': change_filepath}}

        if reference_mesh is True:
            self.reference_mesh = {'filename': change_filepath,
                                   'mesh': read_vtk(change_filepath),
                                   'id': id}
        self.participant_idx += 1

    def add_pre_post_filepaths(self,
                               pre_filepath,
                               post_filepath,
                               participant_identifier=None,
                               reference_mesh=False):
        id = participant_identifier or self.participant_idx
        if id in self.dict_meshes:
            self.dict_meshes[id]['pre'] = {'filename': pre_filepath}
            self.dict_meshes[id]['post'] = {'filename': post_filepath}
        else:
            self.dict_meshes[id] = {'pre': {'filename': pre_filepath},
                                    'post': {'filename': post_filepath}}
        if reference_mesh is True:
            self.reference_mesh = {'filename': pre_filepath,
                                   'mesh': read_vtk(pre_filepath),
                                   'id': id}

        self.participant_idx += 1

    def add_change_data_directly(self, change_data):
        self.added_change_values_directly = True
        self.change_values = change_data

    def calculate_change_mesh(self):
        """
        Place holder function for a future one that will calculate the change mesh (if it doesnt exist).
        :return:
        """

    def compile_data(self):
        """
        Placeholder - classes that inherit must define this
        :return:
        """

    def compute_test_statistics(self):
        """
        Placeholder - classes that inherit must define this
        :return:
        """

    def cluster_test_statistics(self):
        for map_threshold in self.map_threshold:
            clust = Cluster(statistic_mesh=self.test_statistic_maps[self.map_name],
                            statistic_threshold=map_threshold,
                            threshold_type='two_sided',
                            clust_names='cluster',
                            clust_idx=0
                            )
            self.clustered_test_statistic_maps[self.map_name][map_threshold] = clust.get_clusters()

    def compute_mc_thresholds(self):
        """
        Placeholder - classes that inherit must define this
        :return:
        """

    def update(self):
        if self.added_change_values_directly is False:
            self.compile_data()
        print('Finished Loading & Compiling all Data')
        self.compute_test_statistics()
        print('Finished Calculating Test Statistics')
        self.cluster_test_statistics()
        print('Finished Clustering')

        if self.compute_min_cluster_size is True:
            self.compute_mc_thresholds()

    def create_mesh_all_clusters_per_threshold(self):
        self.combined_clustered_meshes = {}
        # Iterate over all map_threshold.
        for map_threshold in self.map_threshold:
            # Add the clusters - they should have the test statistic & the "raw" result - correlation, mean change, etc.
            if len(self.clustered_test_statistic_maps[self.map_name][map_threshold].keys()) == 0:
                self.combined_clustered_meshes[map_threshold] = 'no_significant_clusters'
            else:
                self.combined_clustered_meshes[map_threshold] = combine_clusters_to_one_map(self.clustered_test_statistic_maps[self.map_name][map_threshold])

    def get_combined_clusters(self):
        self.create_mesh_all_clusters_per_threshold()
        return self.combined_clustered_meshes

    def create_mesh_all_sig_clusters_per_threshold(self):
        if self.compute_min_cluster_size is False:
            self.compute_mc_thresholds()

        self.calc_significant_clusters()
        for map_threshold in self.map_threshold:
            # Add the clusters - they should have the test statistic & the "raw" result - correlation, mean change, etc.
            self.combined_sig_clustered_meshes[map_threshold] = combine_clusters_to_one_map(
                self.sig_clusters[self.map_name][map_threshold])

    def get_combined_significant_clusters(self):
        self.create_mesh_all_sig_clusters_per_threshold()
        return self.combined_sig_clustered_meshes

    def create_full_mesh_sig_clusters_per_threshold(self):
        self.create_mesh_all_sig_clusters_per_threshold()

        for map_threshold in self.map_threshold:
            self.sig_clusters_meshes[map_threshold] = transfer_clusters_to_ref_mesh(
                self.reference_mesh['mesh'],
                self.combined_sig_clustered_meshes[map_threshold]
            )

    def get_full_mesh_sig_clusters(self):
        self.create_full_mesh_sig_clusters_per_threshold()
        return self.sig_clusters_meshes


class SimpleTimeDifference(SingleStatisticSPM):
    def __init__(self,
                 *args,
                 map_name='z_statistic',
                 **kwargs
                 ):
        kwargs['map_name'] = map_name
        super().__init__(*args, **kwargs)

    def compile_data(self):
        # ADD LOGIC TO TEST IF CHANGE MESHES EXIST. IF THEY DO NOT, COMPUTE THEM USING FOCUSR.

        if bool(self.reference_mesh) is False:
            if self.find_ref_mesh is True:
                #  Find the template mesh ... this could take a long time.
                raise Exception('Find template mesh not yet implemented')
            else:
                ref_key = sorted(self.dict_meshes.keys())[0]
                if 'change' in self.dict_meshes[ref_key]:
                    self.reference_mesh = self.dict_meshes[ref_key]['change']
                elif 'pre' in self.dict_meshes[ref_key]:
                    self.reference_mesh = self.dict_meshes[ref_key]['pre']
                else:
                    raise Exception('No Pre or Change mesh defined. MUST have at least one of them')
                self.reference_mesh['id'] = ref_key
        self.n_participants = len(self.dict_meshes.keys())
        self.n_points = self.reference_mesh['mesh'].GetNumberOfPoints()
        self.change_values = np.zeros((self.n_participants,
                                       self.n_points))

        for participant_idx, participant_id in enumerate(self.dict_meshes.keys()):
            print('Loading mesh number: {}'.format(participant_idx))
            target_mesh = read_vtk(self.dict_meshes[participant_id]['change']['filename'])
            transform = get_icp_transform(source=target_mesh,
                                          target=self.reference_mesh['mesh'],
                                          reg_mode='similarity')
            target_mesh = apply_transform(source=target_mesh,
                                          transform=transform)
            reg = pyfocusr.Focusr(vtk_mesh_target=target_mesh,
                                  vtk_mesh_source=self.reference_mesh['mesh'],
                                  n_spectral_features=3,
                                  n_extra_spectral=3,
                                  get_weighted_spectral_coords=False,
                                  list_features_to_calc=['curvature'],  # 'curvatures', min_curvature' 'max_curvature'
                                  rigid_reg_max_iterations=100,
                                  non_rigid_alpha=0.01,
                                  non_rigid_beta=50,
                                  non_rigid_n_eigens=100,
                                  non_rigid_max_iterations=self.registration_max_iterations,
                                  rigid_before_non_rigid_reg=False,
                                  projection_smooth_iterations=30,
                                  graph_smoothing_iterations=300,
                                  feature_smoothing_iterations=30,
                                  include_points_as_features=False,
                                  norm_physical_and_spectral=True,
                                  feature_weights=np.diag([.1, .1]),
                                  n_coords_spectral_ordering=10000,
                                  n_coords_spectral_registration=1000,
                                  initial_correspondence_type='kd',
                                  final_correspondence_type='kd')  # 'kd' 'hungarian'
            reg.align_maps()
            reg.get_source_mesh_transformed_weighted_avg()
            ref_mesh_transformed_to_target = reg.weighted_avg_transformed_mesh
            target_change_smoothed_on_ref = transfer_mesh_scalars_get_weighted_average_n_closest(ref_mesh_transformed_to_target,
                                                                                                 target_mesh,
                                                                                                 n=3)
            self.change_values[participant_idx, :] = target_change_smoothed_on_ref
        # Get all non finite values and assign to be zeros.
        self.change_values[np.isnan(self.change_values)] = 0
        self.change_values[np.isinf(self.change_values)] = 0
        self.change_values[np.isneginf(self.change_values)] = 0

    def compute_test_statistics(self):
        n_ppl_with_data_change_per_point = np.sum(self.change_values != 0, axis=0)
        self.idx_no_data = np.where(n_ppl_with_data_change_per_point <
                                    self.percent_participants_with_data_to_include_vertex*self.change_values.shape[0])
        test = test_statistics.OneSampleZTest(self.change_values,
                                              self.reference_mesh['mesh'],
                                              return_new_mesh=True,
                                              idx_not_to_include=self.idx_no_data
                                              )
        test.compute_statistics_per_node()
        test_mesh = test.get_statistics_mesh()
        test_mesh.GetPointData().SetActiveScalars(self.map_name)
        self.test_statistic_maps[self.map_name] = test.get_statistics_mesh()

    def compute_mc_thresholds(self):
        mc_sim = MonteCarloClusterOneSampleTest(self.reference_mesh['mesh'],
                                                self.change_values,  # shape = (participants, pts, other... factors)
                                                method=self.mc_cluster_method,
                                                map_threshold=self.map_threshold,
                                                n_iterations=self.n_monte_carlo_iterations,
                                                idx_not_to_include=self.idx_no_data,
                                                idx_to_include=None)
        mc_sim.update()
        self.threshold_cluster_distribution = mc_sim.get_distribution_of_max_clustersizes()
        self.threshold_cluster_size = mc_sim.get_threshold_clustersize(threshold=self.mc_cluster_extent_significance)
        self.threshold_test_statistic = mc_sim.get_threshold_test_statistic(threshold=self.mc_point_significance)


class SimpleCorrelation(SingleStatisticSPM):
    def __init__(self,
                 *args,
                 map_name='t_statistic',
                 **kwargs
                 ):
        kwargs['map_name'] = map_name
        super().__init__(*args, **kwargs)

    def add_change_filepath(self,
                            *args,
                            secondary_data=None,
                            participant_identifier=None,
                            **kwargs):
        """
        Could potentially have just implemented secondary data to the base version? Even if superfluous?
        :param args:
        :param secondary_data:
        :param participant_identifier:
        :param kwargs:
        :return:
        """
        kwargs['participant_identifier'] = participant_identifier
        super().add_change_filepath(*args, **kwargs)
        id = participant_identifier or (self.participant_idx-1)  # Previous version will have already incremented idx

        if secondary_data is not None:
            self.dict_meshes[id]['change']['secondary_data'] = secondary_data

    def add_pre_post_filepaths(self,
                               *args,
                               pre_secondary_data=None,
                               post_secondary_data=None,
                               participant_identifier=None,
                               **kwargs):
        """
        Could potentially have just implemented secondary data to the base version? Even if superfluous?
        :param args:
        :param pre_secondary_data:
        :param post_secondary_data:
        :param participant_identifier:
        :param kwargs:
        :return:
        """
        kwargs['participant_identifier'] = participant_identifier
        super().add_pre_post_filepaths(*args, **kwargs)
        id = participant_identifier or (self.participant_idx - 1)  # Previous version will have already incremented idx

        if pre_secondary_data is not None:
            self.dict_meshes[id]['pre']['secondary_data'] = pre_secondary_data

        if post_secondary_data is not None:
            self.dict_meshes[id]['post']['secondary_data'] = post_secondary_data

    def compile_data(self):
        # ADD LOGIC TO TEST IF CHANGE MESHES EXIST. IF THEY DO NOT, COMPUTE THEM USING FOCUSR.

        if bool(self.reference_mesh) is False:
            if self.find_ref_mesh is True:
                #  Find the template mesh ... this could take a long time.
                raise Exception('Find template mesh not yet implemented')
            else:
                ref_key = sorted(self.dict_meshes.keys())[0]
                if 'change' in self.dict_meshes[ref_key]:
                    self.reference_mesh = self.dict_meshes[ref_key]['change']
                elif 'pre' in self.dict_meshes[ref_key]:
                    self.reference_mesh = self.dict_meshes[ref_key]['pre']
                else:
                    raise Exception('No Pre or Change mesh defined. MUST have at least one of them')
                self.reference_mesh['id'] = ref_key
        self.n_participants = len(self.dict_meshes.keys())
        self.n_points = self.reference_mesh['mesh'].GetNumberOfPoints()
        self.change_values = np.zeros((self.n_participants,
                                       self.n_points,
                                       2))

        for participant_idx, participant_id in enumerate(self.dict_meshes.keys()):
            print('Loading mesh number: {}'.format(participant_idx))
            target_mesh = read_vtk(self.dict_meshes[participant_id]['change']['filename'])
            transform = get_icp_transform(source=target_mesh,
                                          target=self.reference_mesh['mesh'],
                                          reg_mode='similarity')
            target_mesh = apply_transform(source=target_mesh,
                                          transform=transform)

            reg = pyfocusr.Focusr(vtk_mesh_target=target_mesh,
                                  vtk_mesh_source=self.reference_mesh['mesh'],
                                  n_spectral_features=3,
                                  n_extra_spectral=3,
                                  get_weighted_spectral_coords=False,
                                  list_features_to_calc=['curvature'],  # 'curvatures', min_curvature' 'max_curvature'
                                  rigid_reg_max_iterations=100,
                                  non_rigid_alpha=0.01,
                                  non_rigid_beta=50,
                                  non_rigid_n_eigens=100,
                                  non_rigid_max_iterations=self.registration_max_iterations,
                                  rigid_before_non_rigid_reg=False,
                                  projection_smooth_iterations=30,
                                  graph_smoothing_iterations=300,
                                  feature_smoothing_iterations=30,
                                  include_points_as_features=False,
                                  norm_physical_and_spectral=True,
                                  feature_weights=np.diag([.1, .1]),
                                  n_coords_spectral_ordering=10000,
                                  n_coords_spectral_registration=1000,
                                  initial_correspondence_type='kd',
                                  final_correspondence_type='kd')  # 'kd' 'hungarian'

            reg.align_maps()
            reg.get_source_mesh_transformed_weighted_avg()
            ref_mesh_transformed_to_target = reg.weighted_avg_transformed_mesh
            target_change_smoothed_on_ref = transfer_mesh_scalars_get_weighted_average_n_closest(ref_mesh_transformed_to_target,
                                                                                                 target_mesh,
                                                                                                 n=3)
            if 'secondary_data' in self.dict_meshes[participant_id]['change']:
                data = self.dict_meshes[participant_id]['change']['secondary_data']
                if isinstance(data, (int, float)) and not isinstance(data, bool):
                    self.change_values[participant_idx, :, 1] = data
                elif isinstance(data, (list, np.ndarray)):
                    if (len(data) == 1) or (len(data) == self.n_points):
                        self.change_values[participant_idx, :, 1] = data
                    else:
                        raise Exception('Secondary data of type {} is wrong length, len={},'
                                        'mesh has {} points'.format(type(data), len(data), self.n_points))
                else:
                    raise Exception('Data is type: {}, require: int, float, list, or np.ndarray'.format(type(data)))
            else:
                raise Exception('No secondary data inputted when providing mesh locations. Future work will allow'
                                'secondary data to be on the mesh - this does not exist yet.')
                # To the above warning. Could append data to the mesh with a known array name and then extract.
            self.change_values[participant_idx, :, 0] = target_change_smoothed_on_ref

    def compute_test_statistics(self):
        """
        This is essentially the same as the logic in the SimpleTimeDifference class. Should look at combining
        :return:
        """
        # test to see points with primary outcome only (ignoring secondary for now)
        n_ppl_with_data_change_per_point = np.sum(self.change_values[:, :, 0] != 0, axis=0)
        self.idx_no_data = np.where(n_ppl_with_data_change_per_point <
                                    self.percent_participants_with_data_to_include_vertex*self.n_participants)
        test = test_statistics.CorrelationTTest(self.change_values,
                                                self.reference_mesh['mesh'],
                                                return_new_mesh=True,
                                                idx_not_to_include=self.idx_no_data
                                                )
        test.compute_statistics_per_node()
        test_mesh = test.get_statistics_mesh()
        test_mesh.GetPointData().SetActiveScalars(self.map_name)
        self.test_statistic_maps[self.map_name] = test_mesh

    def compute_mc_thresholds(self):
        """
        This is essentially the same as the logic in the SimpleTimeDifference class. Should look at combining
        :return:
        """
        mc_sim = MonteCarloClusterCorrelationTest(self.reference_mesh['mesh'],
                                                  self.change_values,  # shape = (participants, pts, other... factors)
                                                  method=self.mc_cluster_method,
                                                  map_threshold=self.map_threshold,
                                                  n_iterations=self.n_monte_carlo_iterations,
                                                  idx_not_to_include=self.idx_no_data,
                                                  idx_to_include=None)
        mc_sim.update()
        self.threshold_cluster_distribution = mc_sim.get_distribution_of_max_clustersizes()
        self.threshold_cluster_size = mc_sim.get_threshold_clustersize(threshold=self.mc_cluster_extent_significance)
        self.threshold_test_statistic = mc_sim.get_threshold_test_statistic(threshold=self.mc_point_significance)


def combine_clusters_to_one_map(dict_clusters):
    combine_poly = vtk.vtkAppendPolyData()
    for clus_key in dict_clusters.keys():
        combine_poly.AddInputData(dict_clusters[clus_key]['mesh'])
    combine_poly.Update()
    new_polydata = vtk.vtkPolyData()
    new_polydata.DeepCopy(combine_poly.GetOutput())
    return new_polydata
