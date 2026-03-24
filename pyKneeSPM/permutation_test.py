import numpy as np
from math import factorial, comb
import itertools
from functools import reduce
from sympy.utilities.iterables import multiset_permutations


class PermutationTest(object):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_possible_unique_permutations = 2 ** self.n_participants
        self.sign_permutation_matrix = None
        self.rand_sign_permutations = None

    def get_rand_sign_permutation_matrix(self):
        return self.rand_sign_permutations

    def get_all_unique_sign_permutations(self):
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
        # Create matrix for calculating all possible permutations.
        unique_sign_permutation_matrix = np.zeros((unique_permutations.shape[1],
                                                   self.n_participants,
                                                   self.n_participants))
        for iter_ in range(unique_permutations.shape[1]):
            unique_sign_permutation_matrix[iter_, :, :] = np.diag(unique_permutations[:, iter_])

        return unique_sign_permutation_matrix

    def get_random_sign_permutations(self):
        sign_permutation = np.random.random_sample(self.n_participants * self.n_permutations)
        sign_permutation = np.where(sign_permutation > 0.5, 1.0, -1.0)
        sign_permutation = np.reshape(sign_permutation, (self.n_participants, self.n_permutations))

        sign_permutation_matrix = np.zeros((self.n_permutations, self.n_participants, self.n_participants))
        for iter_ in range(self.n_permutations):
            sign_permutation_matrix[iter_, :, :] = np.diag(sign_permutation[:, iter_])
        return sign_permutation_matrix

    def perform_sign_permutations(self):
        # Test for this could be to calculate all possible permutations up to a certain number (that can be
        # verified manually).
        if self.n_possible_unique_permutations < self.n_permutations:
            self.sign_permutation_matrix = self.get_all_unique_sign_permutations()
        else:
            self.sign_permutation_matrix = self.get_random_sign_permutations()

    def apply_sign_permutatations(self):
        self.rand_sign_permutations = self.sign_permutation_matrix @ self.data

    def update(self):
        self.perform_sign_permutations()
        self.apply_sign_permutatations()


class PermutationCorrelationTTest(PermutationTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_possible_unique_permutations = factorial(self.n_participants)
        self.permutation_matrix = None
        self.rand_permutations = None

    def get_rand_permutation_matrix(self):
        return self.rand_permutations

    def perform_permutations(self):
        n_perms = min(self.n_permutations, self.n_possible_unique_permutations)
        self.permutation_matrix = np.zeros((n_perms, self.n_participants), dtype=int)
        participants = np.arange(self.n_participants)
        self.permutation_matrix[0, :] = np.copy(participants)  # We want to have things in their normal order once
        np.random.shuffle(participants)  # shuffle so dont always get same data (if sample less than possible combos).
        permutations = itertools.permutations(participants)
        for iter_ in range(1, n_perms):
            self.permutation_matrix[iter_, :] = next(permutations)

    def apply_permutatations(self):
        n_perms = min(self.n_permutations, self.n_possible_unique_permutations)
        self.rand_permutations = np.zeros((n_perms,) + self.data.shape)
        self.rand_permutations[:, :, :, 0] = self.data[:, :, 0]

        for iter_ in range(n_perms):
            self.rand_permutations[iter_, :, :, 1] = self.data[self.permutation_matrix[iter_, :], :, 1]

        self.permutation_matrix = None

    def update(self):
        self.perform_permutations()
        self.apply_permutatations()


class PermutationGroupLabels:
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