import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from .permutation_test import *
from .test_statistics import *
from .clustering import Cluster


class MonteCarloThreshold(object):
    """Base class for Monte Carlo cluster-extent inference.

    Builds a null distribution of maximum cluster sizes by repeatedly
    permuting the data, computing a test statistic at every vertex,
    thresholding, extracting clusters, and recording their sizes.
    The (1 - alpha) percentile of this distribution gives the minimum
    cluster size required for family-wise error-corrected significance.

    Parameters
    ----------
    ref_mesh : vtkPolyData
        Reference surface mesh (all participants registered to this).
    data : np.ndarray
        Participant data.  Shape is ``(n_participants, n_vertices, ...)``
        where trailing dimensions depend on the test type.
    method : str
        Permutation strategy — currently only ``'permutation'`` is supported.
    threshold : float
        Cluster-forming threshold applied to the test statistic map.
    n_iterations : int
        Number of Monte Carlo iterations (permutations).
    idx_not_to_include, idx_to_include : array-like, optional
        Vertex indices to exclude / include.  Provide at most one; the
        complement is computed automatically.
    seed : int or np.random.Generator or None, optional
        Seed handed to the permutation scheme, so the null this Monte Carlo
        builds -- and therefore the cluster-extent threshold and every cluster
        p-value derived from it -- is reproducible.  See
        ``permutation_test.DEFAULT_SEED``.

        Note that when ``threshold`` is a list, the calling SPM builds one
        ``MonteCarlo*`` per threshold; each gets the same seed and therefore the
        same permutations, so the thresholds are compared on one realisation of
        the null rather than several.
    """

    def __init__(self,
                 ref_mesh,
                 data,
                 method='permutation',
                 threshold=2.33,
                 n_iterations=1000,
                 idx_not_to_include=None,
                 idx_to_include=None,
                 seed=DEFAULT_SEED):
        self.seed = seed
        self.ref_mesh = ref_mesh
        self.data = data
        self.n_participants, self.n_points = data.shape[:2]
        self.method = method
        self.threshold = threshold
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
    """Monte Carlo inference for a single statistic map.

    Collects per-iteration cluster sizes and max test statistics, then
    computes threshold values at a given alpha level.  Subclasses define
    ``update_permutations()`` and ``perform_monte_carlo()``.
    """

    def __init__(self,
                 *args,
                 map_name='test_map',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.map_name = map_name
        self.cluster_sizes[self.map_name] = {'all': []}
        self.rand_permutations = None
        self.threshold_clustersizes = {}
        self.max_test_statistics[self.map_name] = {'min': [],
                                                   'max': [],
                                                   'abs_max': []}
        self.threshold_test_statistics = {}

    def get_threshold_test_statistic(self, threshold=0.05):
        if threshold not in self.threshold_test_statistics:
            self.calc_threshold_statistic(threshold=threshold)
        return self.threshold_test_statistics[threshold]

    def calc_threshold_statistic(self, threshold=0.05):
        threshold_idx = np.ceil((1 - threshold) * self.n_iterations).astype(int)
        sorted_max_test_statistics = sorted(self.max_test_statistics[self.map_name]['abs_max'])
        self.threshold_test_statistics[threshold] = sorted_max_test_statistics[threshold_idx - 1]

    def get_distribution_of_max_clustersizes(self):
        return sorted(self.cluster_sizes[self.map_name]['max'])

    def get_threshold_clustersize(self, threshold=0.05):
        if threshold not in self.threshold_clustersizes:
            self.calc_threshold_clustersize(threshold=threshold)
        return self.threshold_clustersizes[threshold]

    def calc_threshold_clustersize(self, threshold=0.05):
        threshold_idx = np.ceil((1-threshold) * self.n_iterations).astype(int)
        sorted_max_clustersizes = sorted(self.cluster_sizes[self.map_name]['max'])
        self.threshold_clustersizes[threshold] = sorted_max_clustersizes[threshold_idx-1]

    def find_max_cluster_sizes(self):
        self.cluster_sizes[self.map_name]['max'] = [np.nanmax(x) if len(x) > 0 else 0 for x in self.cluster_sizes[self.map_name]['all']]

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
    """Cluster-extent inference for one-sample z-tests using sign-flip permutations.

    Permutations are applied lazily in batches to avoid materialising all
    permuted datasets at once.

    Parameters
    ----------
    permutation_batch_size : int
        Number of permutations to materialise simultaneously.  Controls
        the memory / vectorisation trade-off.  Peak memory is approximately
        ``batch_size * n_participants * n_vertices * 8 bytes``.
    """

    def __init__(self,
                 *args,
                 map_name='z_statistic',
                 permutation_batch_size=10,
                 **kwargs):
        kwargs['map_name'] = map_name
        super().__init__(*args, **kwargs)
        self.permutation_batch_size = permutation_batch_size
        self.perm_test = None

    def update_permutations(self):
        self.perm_test = PermutationOneSampleZTest(data=self.data_to_simulate,
                                                    n_permutations=self.n_iterations,
                                                    seed=self.seed,
                                                    )
        self.perm_test.update()
        self.n_iterations = self.perm_test.n_permutations

    def perform_monte_carlo(self):
        iter_count = 0
        for batch_start in range(0, self.n_iterations, self.permutation_batch_size):
            batch_end = min(batch_start + self.permutation_batch_size, self.n_iterations)
            batch = self.perm_test.apply_sign_permutation_batch(self.data_to_simulate, batch_start, batch_end)

            for i in range(batch.shape[0]):
                if (self.idx_to_include is not None) & (self.method == 'permutation'):
                    change_array = np.zeros_like(self.data)
                    change_array[:, self.idx_to_include] = batch[i]
                elif (self.idx_to_include is None) & (self.method == 'permutation'):
                    change_array = batch[i]
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

                clust = Cluster(statistic_mesh=z_mesh,
                                statistic_threshold=self.threshold,
                                threshold_type='two_sided',
                                clust_names='cluster',
                                clust_idx=0)
                areas = clust.get_areas()
                self.cluster_sizes[self.map_name]['all'].append(areas)
                iter_count += 1
                print('Completed iteration: {}/{}'.format(iter_count, self.n_iterations))

            del batch


class MonteCarloClusterCorrelationTest(MonteCarloSingleTest):
    """Cluster-extent inference for correlation t-tests using label-shuffle permutations.

    Permutations are applied lazily in batches to avoid materialising all
    permuted datasets at once.

    Parameters
    ----------
    permutation_batch_size : int
        Number of permutations to materialise simultaneously.  Controls
        the memory / vectorisation trade-off.  Peak memory is approximately
        ``batch_size * n_participants * n_vertices * 2 * 8 bytes``.
    """

    def __init__(self,
                 *args,
                 map_name='t_statistic',
                 permutation_batch_size=10,
                 **kwargs):
        kwargs['map_name'] = map_name
        super().__init__(*args, **kwargs)
        self.permutation_batch_size = permutation_batch_size
        self.perm_test = None

    def update_permutations(self):
        self.perm_test = PermutationCorrelationTTest(data=self.data_to_simulate,
                                                     n_permutations=self.n_iterations,
                                                     seed=self.seed)
        self.perm_test.update()
        self.n_iterations = self.perm_test.n_permutations

    def perform_monte_carlo(self):
        iter_count = 0
        for batch_start in range(0, self.n_iterations, self.permutation_batch_size):
            batch_end = min(batch_start + self.permutation_batch_size, self.n_iterations)
            batch = self.perm_test.apply_permutation_batch(self.data_to_simulate, batch_start, batch_end)

            for i in range(batch.shape[0]):
                if (self.idx_to_include is not None) & (self.method == 'permutation'):
                    change_array = np.zeros_like(self.data)
                    change_array[:, self.idx_to_include] = batch[i]
                elif (self.idx_to_include is None) & (self.method == 'permutation'):
                    change_array = batch[i]
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

                clust = Cluster(statistic_mesh=t_mesh,
                                statistic_threshold=self.threshold,
                                threshold_type='two_sided',
                                clust_names='cluster',
                                clust_idx=0)
                areas = clust.get_areas()
                self.cluster_sizes[self.map_name]['all'].append(areas)
                iter_count += 1
                print('Completed iteration: {}/{}'.format(iter_count, self.n_iterations))

            del batch  # free batch memory before next batch


class MonteCarloClusterGroupTest(MonteCarloSingleTest):
    """Cluster-extent inference for group-comparison tests.

    Generates a null distribution by shuffling group labels and re-computing
    the test statistic on each permutation.  Already memory-efficient: only
    the shuffled label vectors are stored, and data is never copied.

    Parameters
    ----------
    group_labels : array-like
        Group label for each participant.
    test_statistic_class : class
        The vertex-wise test statistic class to instantiate per iteration
        (e.g. ``IndependentSamplesTTest``, ``OneWayANOVA``).
    threshold_type : str
        ``'two_sided'`` or ``'one_sided'``, passed to cluster extraction.
    """

    def __init__(self, *args, group_labels, test_statistic_class, threshold_type='two_sided', **kwargs):
        self.group_labels = group_labels
        self.test_statistic_class = test_statistic_class
        self.threshold_type = threshold_type
        self.shuffled_labels = None
        super().__init__(*args, **kwargs)

    def update_permutations(self):
        perm = PermutationGroupLabels(group_labels=self.group_labels,
                                      n_permutations=self.n_iterations,
                                      seed=self.seed)
        perm.update()
        self.shuffled_labels = perm.get_shuffled_labels()
        self.n_iterations = self.shuffled_labels.shape[0]

    def perform_monte_carlo(self):
        for iter_ in range(self.n_iterations):
            stat_test = self.test_statistic_class(
                data=self.data,
                mesh=self.ref_mesh,
                return_new_mesh=True,
                idx_not_to_include=self.idx_not_to_include,
                group_labels=self.shuffled_labels[iter_]
            )
            stat_test.compute_statistics_per_node()
            stat_mesh = stat_test.get_statistics_mesh()
            stat_mesh.GetPointData().SetActiveScalars(self.map_name)

            scalars = vtk_to_numpy(stat_mesh.GetPointData().GetArray(self.map_name))
            self.max_test_statistics[self.map_name]['min'].append(np.nanmin(scalars))
            self.max_test_statistics[self.map_name]['max'].append(np.nanmax(scalars))
            self.max_test_statistics[self.map_name]['abs_max'].append(np.nanmax(np.abs(scalars)))

            clust = Cluster(statistic_mesh=stat_mesh,
                            statistic_threshold=self.threshold,
                            threshold_type=self.threshold_type,
                            clust_names='cluster',
                            clust_idx=0)
            areas = clust.get_areas()
            self.cluster_sizes[self.map_name]['all'].append(areas)
            print('Completed iteration: {}/{}'.format(iter_ + 1, self.n_iterations))


class MonteCarloClusterTwoSampleTest(MonteCarloClusterGroupTest):
    """Cluster-extent inference for independent-samples t-tests.

    Convenience subclass that wires ``IndependentSamplesTTest`` as the
    test statistic and uses two-sided thresholding.
    """

    def __init__(self, *args, map_name='t_statistic', threshold_type='two_sided', **kwargs):
        kwargs['map_name'] = map_name
        super().__init__(*args, test_statistic_class=IndependentSamplesTTest, threshold_type=threshold_type, **kwargs)


class MonteCarloClusterANOVA(MonteCarloClusterGroupTest):
    """Cluster-extent inference for one-way ANOVA F-tests.

    Convenience subclass that wires ``OneWayANOVA`` as the test statistic
    and uses one-sided thresholding (F is always positive).
    """

    def __init__(self, *args, map_name='f_statistic', threshold_type='one_sided', **kwargs):
        kwargs['map_name'] = map_name
        super().__init__(*args, test_statistic_class=OneWayANOVA, threshold_type=threshold_type, **kwargs)
