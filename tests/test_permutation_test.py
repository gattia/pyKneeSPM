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


class TestPermutationCorrelationTTestNonDegenerate:
    """Regression tests for the 2026-07-26 permutation bug.

    The old implementation shuffled the participant vector ONCE and then pulled successive
    draws from ``itertools.permutations``, which enumerates lexicographically by position.
    Consecutive draws therefore differed only in the trailing few elements: at n=694 with
    10,000 draws, 686 of 694 positions were frozen and no position ever took more than 9
    distinct values.  The null distribution collapsed to ~1 effective random pairing.
    """

    def test_no_position_is_frozen_across_draws(self):
        """Every participant position must move across the permutation draws."""
        np.random.seed(0)
        n_participants, n_perms = 60, 500
        data = np.random.randn(n_participants, 4, 2)

        perm = PermutationCorrelationTTest(data=data, n_permutations=n_perms)
        perm.update()
        matrix = perm.get_permutation_indices()

        # Row 0 is the identity by design; the permutations proper are rows 1..N-1.
        frozen = [j for j in range(n_participants)
                  if len(np.unique(matrix[1:, j])) == 1]
        assert frozen == [], f'positions frozen across all draws: {frozen}'

    def test_every_position_explores_most_participants(self):
        """Each column should see nearly every participant index, not a handful.

        Under the old scheme this was capped at ~9 regardless of n.  With independent
        draws the expected number of distinct values in a column over ``n_perms`` draws is
        ``n * (1 - (1 - 1/n) ** n_perms)``, i.e. essentially n once n_perms >> n.
        """
        np.random.seed(0)
        n_participants, n_perms = 60, 500
        data = np.random.randn(n_participants, 4, 2)

        perm = PermutationCorrelationTTest(data=data, n_permutations=n_perms)
        perm.update()
        matrix = perm.get_permutation_indices()

        distinct_per_position = np.array([len(np.unique(matrix[:, j]))
                                          for j in range(n_participants)])
        assert distinct_per_position.min() >= 0.9 * n_participants, (
            f'min distinct values per position = {distinct_per_position.min()} '
            f'(expected ~{n_participants})')

    def test_participant_position_is_uniform(self):
        """A given participant should land in every position with roughly equal frequency."""
        np.random.seed(0)
        n_participants, n_perms = 40, 4000
        data = np.random.randn(n_participants, 3, 2)

        perm = PermutationCorrelationTTest(data=data, n_permutations=n_perms)
        perm.update()
        matrix = perm.get_permutation_indices()[1:]  # drop the identity row

        positions = np.array([int(np.flatnonzero(row == 0)[0]) for row in matrix])
        expected_mean = (n_participants - 1) / 2.0
        expected_sd = np.sqrt((n_participants ** 2 - 1) / 12.0)
        # generous tolerance — this is a degeneracy check, not a strict RNG test
        assert abs(positions.mean() - expected_mean) < 0.15 * n_participants
        assert positions.std() > 0.7 * expected_sd

    def test_identity_row_first(self):
        """Row 0 must be the identity so the observed statistic is part of the null."""
        np.random.seed(0)
        for n_participants, n_perms in ((4, 10000), (30, 100)):
            data = np.random.randn(n_participants, 3, 2)
            perm = PermutationCorrelationTTest(data=data, n_permutations=n_perms)
            perm.update()
            np.testing.assert_array_equal(perm.get_permutation_indices()[0],
                                          np.arange(n_participants))

    def test_every_row_is_a_valid_permutation(self):
        np.random.seed(0)
        n_participants, n_perms = 25, 300
        data = np.random.randn(n_participants, 3, 2)
        perm = PermutationCorrelationTTest(data=data, n_permutations=n_perms)
        perm.update()
        matrix = perm.get_permutation_indices()
        for i, row in enumerate(matrix):
            np.testing.assert_array_equal(np.sort(row), np.arange(n_participants),
                                          err_msg=f'row {i} is not a permutation')

    @pytest.mark.parametrize('n_participants', [3, 5, 6])
    def test_exhaustive_enumeration_is_complete_and_unique(self, n_participants):
        """When n! <= n_permutations every distinct ordering appears exactly once."""
        np.random.seed(0)
        data = np.random.randn(n_participants, 3, 2)
        perm = PermutationCorrelationTTest(data=data, n_permutations=10000)
        perm.update()
        matrix = perm.get_permutation_indices()

        assert matrix.shape == (factorial(n_participants), n_participants)
        assert perm.n_permutations == factorial(n_participants)
        assert len(np.unique(matrix, axis=0)) == factorial(n_participants)

    def test_random_branch_used_when_factorial_exceeds_request(self):
        """n=8 has 8! = 40,320 > 10,000, so the random branch must be taken."""
        np.random.seed(0)
        data = np.random.randn(8, 3, 2)
        perm = PermutationCorrelationTTest(data=data, n_permutations=10000)
        perm.update()
        assert perm.n_permutations == 10000
        assert perm.get_permutation_indices().shape == (10000, 8)
