import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from .permutation_test import *
from .test_statistics import *
from .clustering import Cluster


class MonteCarloThreshold(object):
    def __init__(self,
                 ref_mesh,
                 data,  #M = participants, N = vertices, remaining dimensions may be for more complicated designs
                 method='permutation',
                 map_threshold=2.33,
                 n_iterations=1000,
                 idx_not_to_include=None,
                 idx_to_include=None):
        self.ref_mesh = ref_mesh
        self.data = data
        self.n_participants, self.n_points = data.shape[:2]
        self.method = method
        self.map_threshold = map_threshold if type(map_threshold) in (list, tuple) else [map_threshold, ]
        self.n_iterations = n_iterations
        # Handle the case where either just idx_not_to_include or idx_to_include is provided (and not the other).
        if (idx_not_to_include is None) & (idx_to_include is not None):
            self.idx_not_to_include = np.delete(np.arange(self.n_points), idx_to_include)
        else:
            self.idx_not_to_include = idx_not_to_include
        if (idx_to_include is None) & (idx_not_to_include is not None):
            self.idx_to_include = np.delete(np.arange(self.n_points), idx_not_to_include)
        else:
            self.idx_to_include = idx_to_include
        # Test if we need to subsample the points on the mesh.
        if self.idx_to_include is not None:
            self.data_to_simulate = self.data[:, self.idx_to_include]
        else:
            self.data_to_simulate = self.data[:]

        self.cluster_sizes = {}  # dictionary to collect cluster sizes from potentially multiple maps.
        self.max_test_statistics = {}


class MonteCarloSingleTest(MonteCarloThreshold):
    def __init__(self,
                 *args,
                 map_name='test_map',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.map_name = map_name
        self.rand_permutations = None
        self.threshold_clustersizes = {}
        self.max_test_statistics[self.map_name] = {'min': [],
                                                   'max': [],
                                                   'abs_max': []}
        self.cluster_sizes[self.map_name] = {'all': {},
                                             'max': {}}
        for map_threshold in self.map_threshold:
            self.cluster_sizes[self.map_name]['all'][map_threshold] = []
        self.threshold_test_statistics = {}

    def get_threshold_test_statistic(self, threshold=0.05):
        if threshold not in self.threshold_test_statistics:
            self.calc_threshold_statistic(threshold=threshold)
        return self.threshold_test_statistics[threshold]

    def calc_threshold_statistic(self, threshold=0.05):
        threshold_idx = np.ceil((1 - threshold) * self.n_iterations).astype(np.int)
        sorted_max_test_statistics = sorted(self.max_test_statistics[self.map_name]['abs_max'])
        self.threshold_test_statistics[threshold] = sorted_max_test_statistics[threshold_idx - 1]

    def get_distribution_of_max_clustersizes(self):
        return self.cluster_sizes[self.map_name]['max']

    def get_distribution_of_all_clustersizes(self):
        return self.cluster_sizes[self.map_name]['all']

    def get_threshold_clustersize(self, threshold=0.05):
        if threshold not in self.threshold_clustersizes:
            self.threshold_clustersizes[threshold] = {}
            self.calc_threshold_clustersize(threshold=threshold)
        return self.threshold_clustersizes[threshold]

    def calc_threshold_clustersize(self, threshold=0.05):
        threshold_idx = np.ceil((1-threshold) * self.n_iterations).astype(np.int)
        for map_threshold in self.map_threshold:
            sorted_max_clustersizes = sorted(self.cluster_sizes[self.map_name]['max'][map_threshold])
            self.threshold_clustersizes[threshold][map_threshold] = sorted_max_clustersizes[threshold_idx]

    def find_max_cluster_sizes(self):
        for map_threshold in self.map_threshold:
            self.cluster_sizes[self.map_name]['max'][map_threshold] = sorted([np.nanmax(x) if len(x) > 0 else 0 for x in self.cluster_sizes[self.map_name]['all'][map_threshold]])

    def update(self):
        if self.method == 'permutation':
            self.update_permutations()
        elif self.method == 'rft':
            raise Exception('rft methods not yet instantiated')

        self.perform_monte_carlo()
        self.find_max_cluster_sizes()

    def update_permutations(self):
        """
        Place holder function - the functions that inherit from this should define the permutaitons.
        :return:
        """
        pass

    def perform_monte_carlo(self):
        """
        Place holder function - the functions that inherit from this should define the monte carlo methods.
        :return:
        """
        pass


class MonteCarloClusterOneSampleTest(MonteCarloSingleTest):
    def __init__(self,
                 *args,
                 map_name='z_statistic',
                 **kwargs):
        kwargs['map_name'] = map_name
        super().__init__(*args, **kwargs)

    def update_permutations(self):
        perm_test = PermutationOneSampleZTest(data=self.data_to_simulate,
                                              n_permutations=self.n_iterations,
                                              )
        perm_test.update()
        self.rand_permutations = perm_test.get_rand_sign_permutation_matrix()
        self.n_iterations = self.rand_permutations.shape[0]  # For the cases where we can compute all of them and they are less than the original self.n_iterations

    def perform_monte_carlo(self):
        for iter_ in range(self.n_iterations):

            if (self.idx_to_include is not None) & (self.method == 'permutation'):
                change_array = np.zeros_like(self.data)
                change_array[:, self.idx_to_include] = self.rand_permutations[iter_, :, :]
            elif (self.idx_to_include is None) & (self.method == 'permutation'):
                change_array = self.rand_permutations[iter_, :, :]
            elif self.method == 'rft':
                raise Exception('rft (random field theory) computation of cluster sizes not implemented, yet.')

            z_test = OneSampleZTest(change_array,
                                    self.ref_mesh,
                                    return_new_mesh=True,
                                    idx_not_to_include=self.idx_not_to_include)
            z_test.compute_statistics_per_node()
            z_mesh = z_test.get_statistics_mesh()
            z_mesh.GetPointData().SetActiveScalars(self.map_name)

            z_scalars = vtk_to_numpy(z_mesh.GetPointData().GetArray(self.map_name))
            self.max_test_statistics[self.map_name]['min'].append(np.nanmin(z_scalars))
            self.max_test_statistics[self.map_name]['max'].append(np.nanmax(z_scalars))
            self.max_test_statistics[self.map_name]['abs_max'].append(np.nanmax(np.abs(z_scalars)))

            for map_threshold in self.map_threshold:
                clust = Cluster(statistic_mesh=z_mesh,
                                statistic_threshold=map_threshold,
                                threshold_type='two_sided',
                                clust_names='cluster',
                                clust_idx=0)
                areas = clust.get_areas()
                self.cluster_sizes[self.map_name]['all'][map_threshold].append(areas)
            print('Completed iteration: {}/{}'.format(iter_+1, self.n_iterations))


class MonteCarloClusterCorrelationTest(MonteCarloSingleTest):
    def __init__(self,
                 *args,
                 map_name='t_statistic',
                 **kwargs):
        kwargs['map_name'] = map_name
        super().__init__(*args, **kwargs)

    def update_permutations(self):
        perm_test = PermutationCorrelationTTest(data=self.data_to_simulate,
                                                n_permutations=self.n_iterations)
        perm_test.update()
        self.rand_permutations = perm_test.get_rand_permutation_matrix()
        # For the cases where we can compute all of them and they are less than the original self.n_iterations
        self.n_iterations = self.rand_permutations.shape[0]

    def perform_monte_carlo(self):
        for iter_ in range(self.n_iterations):
            if (self.idx_to_include is not None) & (self.method == 'permutation'):
                change_array = np.zeros_like(self.data)
                change_array[:, self.idx_to_include] = self.rand_permutations[iter_, :, :]
            elif (self.idx_to_include is None) & (self.method == 'permutation'):
                change_array = self.rand_permutations[iter_, :, :]
            elif self.method == 'rft':
                raise Exception('rft (random field theory) computation of cluster sizes not implemented, yet.')

            t_test = CorrelationTTest(change_array,
                                      self.ref_mesh,
                                      return_new_mesh=True,
                                      idx_not_to_include=self.idx_not_to_include)
            t_test.compute_statistics_per_node()
            t_mesh = t_test.get_statistics_mesh()
            t_mesh.GetPointData().SetActiveScalars(self.map_name)

            t_scalars = vtk_to_numpy(t_mesh.GetPointData().GetArray(self.map_name))
            self.max_test_statistics[self.map_name]['min'].append(np.nanmin(t_scalars))
            self.max_test_statistics[self.map_name]['max'].append(np.nanmax(t_scalars))
            self.max_test_statistics[self.map_name]['abs_max'].append(np.nanmax(np.abs(t_scalars)))

            for map_threshold in self.map_threshold:
                clust = Cluster(statistic_mesh=t_mesh,
                                statistic_threshold=map_threshold,
                                threshold_type='two_sided',
                                clust_names='cluster',
                                clust_idx=0)
                areas = clust.get_areas()
                self.cluster_sizes[self.map_name]['all'][map_threshold].append(areas)
            print('Completed iteration: {}/{}'.format(iter_+1, self.n_iterations))



