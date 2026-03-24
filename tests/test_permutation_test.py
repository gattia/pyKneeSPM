import numpy as np
import pytest
from math import factorial

from pyKneeSPM.permutation_test import PermutationOneSampleZTest, PermutationCorrelationTTest


class TestPermutationOneSampleZTest:
    def test_all_unique_permutations_small_n(self):
        """With n=3 participants, 2^3=8 unique sign permutations should be enumerable."""
        np.random.seed(42)
        n_participants, n_vertices = 3, 10
        data = np.random.randn(n_participants, n_vertices)

        perm = PermutationOneSampleZTest(data=data, n_permutations=100)
        assert perm.n_possible_unique_permutations == 8

        # Since 8 < 100, it should compute all unique permutations
        perm.perform_sign_permutations()
        vectors = perm.get_sign_vectors()
        assert vectors.shape == (8, n_participants)

    def test_sign_vectors_are_plus_minus_one(self):
        """Each sign vector should have entries in {-1, +1}."""
        np.random.seed(42)
        data = np.random.randn(3, 10)
        perm = PermutationOneSampleZTest(data=data, n_permutations=100)
        perm.perform_sign_permutations()
        vectors = perm.get_sign_vectors()

        for i in range(vectors.shape[0]):
            assert np.all((vectors[i] == 1) | (vectors[i] == -1))

    def test_unique_permutations_are_unique(self):
        """All 2^n permutations should be distinct."""
        np.random.seed(42)
        data = np.random.randn(3, 5)
        perm = PermutationOneSampleZTest(data=data, n_permutations=100)
        perm.perform_sign_permutations()
        vectors = perm.get_sign_vectors()

        unique_vectors = np.unique(vectors, axis=0)
        assert len(unique_vectors) == 8

    def test_sign_flip_applied_correctly(self):
        """Verify applying sign permutations flips data correctly."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 participants, 2 vertices
        perm = PermutationOneSampleZTest(data=data, n_permutations=100)
        perm.update()
        result = perm.apply_sign_permutation_batch(data, 0, perm.n_permutations)

        # result shape: (n_perms, n_participants, n_vertices)
        assert result.shape[1] == 2
        assert result.shape[2] == 2

        # Each permuted dataset should have same absolute values as original
        for i in range(result.shape[0]):
            np.testing.assert_array_equal(np.abs(result[i]), np.abs(data))

    def test_random_permutations_shape(self):
        """When n is large, random permutations have correct shape."""
        np.random.seed(42)
        n_participants, n_vertices = 20, 15
        data = np.random.randn(n_participants, n_vertices)
        n_perms = 50

        perm = PermutationOneSampleZTest(data=data, n_permutations=n_perms)
        perm.update()
        result = perm.apply_sign_permutation_batch(data, 0, perm.n_permutations)

        assert result.shape == (n_perms, n_participants, n_vertices)


class TestPermutationCorrelationTTest:
    def test_all_permutations_small_n(self):
        """With n=3 participants, 3!=6 permutations should be enumerable."""
        np.random.seed(42)
        n_participants, n_vertices = 3, 10
        data = np.random.randn(n_participants, n_vertices, 2)

        perm = PermutationCorrelationTTest(data=data, n_permutations=100)
        assert perm.n_possible_unique_permutations == 6

        perm.perform_permutations()
        # permutation_matrix shape: (n_perms, n_participants)
        assert perm.permutation_matrix.shape == (6, 3)

    def test_primary_variable_unchanged(self):
        """Column 0 (primary variable) should be unchanged across all permutations."""
        np.random.seed(42)
        n_participants, n_vertices = 3, 10
        data = np.random.randn(n_participants, n_vertices, 2)

        perm = PermutationCorrelationTTest(data=data, n_permutations=100)
        perm.update()
        result = perm.apply_permutation_batch(data, 0, perm.n_permutations)

        # result shape: (n_perms, n_participants, n_vertices, 2)
        for i in range(result.shape[0]):
            np.testing.assert_array_equal(result[i, :, :, 0], data[:, :, 0])

    def test_secondary_variable_is_valid_reordering(self):
        """Each permutation of the secondary variable should be a valid reordering of participants."""
        np.random.seed(42)
        n_participants, n_vertices = 3, 10
        data = np.random.randn(n_participants, n_vertices, 2)

        perm = PermutationCorrelationTTest(data=data, n_permutations=100)
        perm.update()
        result = perm.apply_permutation_batch(data, 0, perm.n_permutations)

        original_secondary = data[:, :, 1]
        for i in range(result.shape[0]):
            permuted_secondary = result[i, :, :, 1]
            # Each row in permuted_secondary should match some row in original_secondary
            # Check that the set of rows is the same (just reordered)
            found = np.zeros(n_participants, dtype=bool)
            for p in range(n_participants):
                for q in range(n_participants):
                    if np.allclose(permuted_secondary[p], original_secondary[q]):
                        found[p] = True
                        break
            assert np.all(found), f"Permutation {i} has invalid reordering"

    def test_permutation_count(self):
        """Should get exactly min(n_permutations, n!) permutations."""
        np.random.seed(42)
        data = np.random.randn(3, 5, 2)

        perm = PermutationCorrelationTTest(data=data, n_permutations=100)
        perm.update()

        assert perm.n_permutations == 6  # 3! = 6 < 100
        assert perm.get_permutation_indices().shape == (6, 3)

    def test_larger_n_random_permutations(self):
        """With larger n, only n_permutations are generated."""
        np.random.seed(42)
        n_participants, n_vertices = 10, 5
        data = np.random.randn(n_participants, n_vertices, 2)
        n_perms = 20

        perm = PermutationCorrelationTTest(data=data, n_permutations=n_perms)
        perm.update()
        result = perm.apply_permutation_batch(data, 0, perm.n_permutations)

        assert result.shape == (n_perms, n_participants, n_vertices, 2)
