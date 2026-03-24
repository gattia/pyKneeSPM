import numpy as np
from math import factorial, comb
import itertools
from functools import reduce
from sympy.utilities.iterables import multiset_permutations


class PermutationTest(object):
    """Base class for permutation schemes used in non-parametric inference.

    Subclasses generate permutation indices or sign vectors lazily — the full
    permuted data is never materialised all at once.  Instead, callers request
    batches via ``apply_*_batch()`` methods, keeping peak memory proportional
    to ``batch_size * single_permutation_size``.

    Parameters
    ----------
    data : np.ndarray
        Input data array.  Shape depends on the test type:
        ``(n_participants, n_vertices)`` for one-sample tests,
        ``(n_participants, n_vertices, 2)`` for correlation tests.
    n_permutations : int
        Requested number of permutations.  If the total number of unique
        permutations is smaller, all unique permutations are generated instead.
    idx_to_include : array-like, optional
        Vertex indices to include.  When provided, only these vertices are
        used during permutation generation (the original full data is kept
        in ``self.orig_data``).
    """

    def __init__(self,
                 data,
                 n_permutations,
                 idx_to_include=None):
        if idx_to_include is not None:
            self.orig_data = data
            self.data = data[:, idx_to_include]
            self.idx_to_include = idx_to_include
        else:
            self.data = data
            self.orig_data = None
        self.n_permutations = n_permutations
        self.n_participants, self.n_points = self.data.shape[:2]


class PermutationOneSampleZTest(PermutationTest):
    """Sign-flip permutations for one-sample tests.

    Each permutation randomly flips the sign of each participant's data
    (multiply by +1 or -1), yielding 2^n possible unique permutations.

    Only the sign vectors are stored — shape ``(n_perms, n_participants)`` —
    not the full permuted datasets.  Use ``apply_sign_permutation_batch()``
    to materialise a batch of permuted data on demand.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_possible_unique_permutations = 2 ** self.n_participants
        self.sign_vectors = None  # (n_perms, n_participants) of +1/-1

    def get_sign_vectors(self):
        return self.sign_vectors

    def get_all_unique_sign_vectors(self):
        n_tests = self.n_possible_unique_permutations * 10
        unique_permutations = np.ones((self.n_participants, 1))
        while unique_permutations.shape[1] < self.n_possible_unique_permutations:

            sign_permutation = np.random.random_sample(self.n_participants * n_tests)
            sign_permutation[sign_permutation > 0.5] = 1
            sign_permutation[sign_permutation < 1] = -1
            sign_permutation = np.reshape(sign_permutation, (self.n_participants,
                                                         n_tests))
            # there is a chance that we dont get all of the unique permutations.
            # could run this recursively? Or on a while loop?
            sign_permutation = np.concatenate((sign_permutation, unique_permutations), axis=1)
            unique_permutations = np.unique(sign_permutation, axis=1)
        # Return as (n_perms, n_participants) — just the sign vectors, not diagonal matrices
        return unique_permutations.T

    def get_random_sign_vectors(self):
        sign_permutation = np.random.random_sample(self.n_participants * self.n_permutations)
        sign_permutation = np.where(sign_permutation > 0.5, 1.0, -1.0)
        # Return as (n_perms, n_participants)
        return np.reshape(sign_permutation, (self.n_permutations, self.n_participants))

    def perform_sign_permutations(self):
        # Test for this could be to calculate all possible permutations up to a certain number (that can be
        # verified manually).
        if self.n_possible_unique_permutations < self.n_permutations:
            self.sign_vectors = self.get_all_unique_sign_vectors()
        else:
            self.sign_vectors = self.get_random_sign_vectors()
        self.n_permutations = self.sign_vectors.shape[0]

    def apply_sign_permutation_batch(self, data, start, end):
        """Materialise a batch of sign-flipped datasets.

        Equivalent to ``diag(signs) @ data`` for each permutation, but uses
        element-wise broadcasting instead of materialising diagonal matrices.

        Parameters
        ----------
        data : np.ndarray, shape (n_participants, n_vertices)
            The original (unpermuted) data.
        start, end : int
            Slice indices into ``self.sign_vectors`` selecting which
            permutations to apply.

        Returns
        -------
        np.ndarray, shape (end - start, n_participants, n_vertices)
        """
        # Broadcasting: (batch, n_part, 1) * (n_part, n_vert) -> (batch, n_part, n_vert)
        return self.sign_vectors[start:end, :, np.newaxis] * data[np.newaxis, :, :]

    def update(self):
        self.perform_sign_permutations()


class PermutationCorrelationTTest(PermutationTest):
    """Participant-reordering permutations for correlation tests.

    For correlation between a surface scalar (column 0) and a secondary
    variable (column 1), the null hypothesis is broken by shuffling which
    participant's secondary variable is paired with which surface data.

    Only the index matrix is stored — shape ``(n_perms, n_participants)`` —
    not the full permuted datasets.  Use ``apply_permutation_batch()`` to
    materialise a batch of permuted data on demand.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_possible_unique_permutations = factorial(self.n_participants)
        self.permutation_matrix = None

    def get_permutation_indices(self):
        return self.permutation_matrix

    def perform_permutations(self):
        n_perms = min(self.n_permutations, self.n_possible_unique_permutations)
        self.permutation_matrix = np.zeros((n_perms, self.n_participants), dtype=int)
        participants = np.arange(self.n_participants)
        self.permutation_matrix[0, :] = np.copy(participants)  # We want to have things in their normal order once
        np.random.shuffle(participants)  # shuffle so dont always get same data (if sample less than possible combos).
        permutations = itertools.permutations(participants)
        for iter_ in range(1, n_perms):
            self.permutation_matrix[iter_, :] = next(permutations)
        self.n_permutations = n_perms

    def apply_permutation_batch(self, data, start, end):
        """Materialise a batch of permuted datasets.

        Column 0 (surface scalar) is kept fixed across all permutations;
        column 1 (secondary variable) is reordered according to the
        stored permutation indices.

        Parameters
        ----------
        data : np.ndarray, shape (n_participants, n_vertices, 2)
            The original (unpermuted) data.
        start, end : int
            Slice indices into ``self.permutation_matrix`` selecting which
            permutations to apply.

        Returns
        -------
        np.ndarray, shape (end - start, n_participants, n_vertices, 2)
        """
        batch_indices = self.permutation_matrix[start:end]
        batch_size = end - start
        permuted = np.tile(data, (batch_size, 1, 1, 1))  # (batch, n_subj, n_vert, 2)
        permuted[:, :, :, 1] = data[batch_indices, :, 1]  # fancy index broadcasts over batch
        return permuted

    def update(self):
        self.perform_permutations()


class PermutationGroupLabels:
    """Group-label shuffling for group-comparison tests (two-sample t, ANOVA).

    Generates permuted group-label vectors by shuffling participant-to-group
    assignments.  When the number of unique permutations (multinomial
    coefficient) is small enough, all are enumerated exhaustively via
    ``sympy.utilities.iterables.multiset_permutations``; otherwise random
    shuffles are drawn.

    Only the label vectors are stored — shape ``(n_perms, n_participants)`` —
    the data itself is never copied or reordered here.

    Parameters
    ----------
    group_labels : array-like
        Group label for each participant (e.g. ``[0, 0, 1, 1, 1]``).
    n_permutations : int
        Requested number of permutations.
    """

    def __init__(self, group_labels, n_permutations):
        self.group_labels = np.asarray(group_labels)
        self.n_participants = len(self.group_labels)
        self.n_permutations = n_permutations
        self.shuffled_labels = None

        # Compute number of unique permutations: multinomial coefficient n! / (n1! * n2! * ... * nk!)
        unique, counts = np.unique(self.group_labels, return_counts=True)
        self.n_possible_unique_permutations = factorial(self.n_participants) // reduce(
            lambda a, b: a * b, (factorial(c) for c in counts)
        )

    def perform_permutations(self):
        if self.n_possible_unique_permutations <= self.n_permutations:
            # Generate all unique permutations using sympy's multiset_permutations.
            # This is O(unique permutations), not O(n!) like itertools.permutations
            # with deduplication. For example, 14 participants in 2 groups of 7 has
            # C(14,7)=3,432 unique permutations but 14!=87 billion total — itertools
            # would hang, sympy completes in ~20ms.
            self.shuffled_labels = np.array(list(multiset_permutations(self.group_labels.tolist())))
            self.n_permutations = len(self.shuffled_labels)
        else:
            self.shuffled_labels = np.zeros((self.n_permutations, self.n_participants), dtype=self.group_labels.dtype)
            for i in range(self.n_permutations):
                self.shuffled_labels[i] = np.random.permutation(self.group_labels)

    def get_shuffled_labels(self):
        return self.shuffled_labels

    def update(self):
        self.perform_permutations()