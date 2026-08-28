# Permutation degeneracy in `PermutationCorrelationTTest` — bug, evidence, fix, validation

**Status:** FIXED 2026-07-26
**Severity:** SEVERE — every cluster-extent p-value and cluster-size threshold produced by
`SimpleCorrelation` (and anything else routed through `MonteCarloClusterCorrelationTest`) before
this date is invalid.
**Scope:** correlation branch only. `PermutationGroupLabels` (two-sample t-test, one-way ANOVA) and
`PermutationOneSampleZTest` were already correct and are **not** implicated.
**Downstream consumer that found it:** `comak_gait_simulation`, Paper-1 Figures 3/5 (pressure↔thinning
correlations) and 6/7 (pressure ~ KAM / body-mass scalar regressions). Their re-run is tracked in
that repo at `.claude/plans/PAPER_SPM_FIX_July.26.2026.md`.

---

## 1. The bug

`pyKneeSPM/permutation_test.py::PermutationCorrelationTTest.perform_permutations` (pre-fix):

```python
def perform_permutations(self):
    n_perms = min(self.n_permutations, self.n_possible_unique_permutations)
    self.permutation_matrix = np.zeros((n_perms, self.n_participants), dtype=int)
    participants = np.arange(self.n_participants)
    self.permutation_matrix[0, :] = np.copy(participants)   # identity — fine
    np.random.shuffle(participants)                         # ONE random shuffle...
    permutations = itertools.permutations(participants)     # ...then enumerate from it
    for iter_ in range(1, n_perms):
        self.permutation_matrix[iter_, :] = next(permutations)
    self.n_permutations = n_perms
```

The intent was clearly "shuffle so we don't always start from the same place." The defect is that
`itertools.permutations` is an **enumerator, not a sampler**: it walks the permutations of its input
in lexicographic order *by position*, so consecutive outputs differ only in a short suffix. Drawing
`N` consecutive items therefore only ever moves the last `k` positions, where `k` is the smallest
integer with `k! >= N`.

At the sizes this library is used for:

| n (participants) | n_permutations | positions that ever move | positions FROZEN across all draws |
|---|---|---|---|
| 9   | 10,000 | 8 | 1 / 9 |
| 222 | 10,000 | 8 | **214 / 222** |
| 694 | 10,000 | 8 | **686 / 694** |

Because `8! = 40,320 > 10,000 >= 7! = 5,040`, exactly the last 8 positions vary regardless of `n`.
Every participant outside that window keeps the pairing it got from the single `np.random.shuffle`.

**Consequence.** The permutation null is not "10,000 random re-pairings of the surface data with the
secondary variable" — it is *one* random re-pairing, jittered in 8 slots. The 95th percentile of the
resulting max-cluster distribution is a single random draw, and it lands wherever that one shuffle
happened to put it. The test is anti-conservative on average (a lucky shuffle yields a low null and
a low threshold) but the direction is not guaranteed run to run.

The identity row (row 0) was already present and is correct — including the observed statistic in the
null is the standard construction and gives the proper `p >= 1/n_perms` floor.

---

## 2. Empirical evidence

### 2.1 Direct inspection of the permutation matrix

```
=== PRE-FIX (rows 1..N-1, i.e. the actual permutation draws) ===
n=   9  n_perms=10000  positions CONSTANT across draws:   1/9    unique values/position: max=9 (ideal 9)
n= 222  n_perms=10000  positions CONSTANT across draws: 214/222  unique values/position: max=9 (ideal 222)
n= 694  n_perms=10000  positions CONSTANT across draws: 686/694  unique values/position: max=9 (ideal 694)
```

Note the second column: **no position ever took more than 9 distinct values**, at any n. With
independent draws each column should see essentially every participant index.

### 2.2 Signature in the shipped results (comak_gait_simulation, pre-fix)

- **Every** cluster p-value in all 48 correlation runs (3 cohorts × 16 runs) is pinned at the
  `1/10000 = 0.0001` floor — i.e. no permuted dataset ever produced a cluster as large as the
  observed one. All 38 Fig-7 scalar regressions sit at their own floor (`1/2000 = 0.0005`).
- Cluster-extent thresholds swing **26×** across runs (0.0019–0.0491) and **6×** between two runs at
  the identical n=694.
- By contrast the correctly-permuted ANOVA branch, on the same meshes and same MC count, sits
  tightly at 0.0116–0.0128 across all six runs — exactly what a stable null looks like.

### 2.3 A/B experiment: same data, same MC count, repeated runs

`femur`, cumulative pressure vs thinning, `kl2` subset (n=222, 20,000 vertices), `N_MC=500`,
3 repetitions per scheme, cluster-forming threshold t=2.33:

| scheme | rep | cluster-extent threshold | null mean | null SD | n sig clusters | total sig area |
|---|---|---|---|---|---|---|
| OLD | 0 | 0.002337 | 0.00172 | 0.00340 | 7 | 0.1459 |
| OLD | 1 | 0.018725 | 0.01746 | 0.00283 | 2 | 0.0978 |
| OLD | 2 | 0.006959 | 0.00371 | 0.00383 | 6 | 0.1421 |
| **NEW** | 0 | 0.024630 | 0.00962 | 0.00806 | 1 | 0.0767 |
| **NEW** | 1 | 0.024307 | 0.00968 | 0.00876 | 1 | 0.0767 |
| **NEW** | 2 | 0.026296 | 0.00984 | 0.00921 | 1 | 0.0767 |

**Threshold spread across three identical runs: OLD = 8.01× · NEW = 1.08×.**

The OLD scheme's threshold moves 8× on identical data and its null *mean* moves 10× — the null level
is set by whichever single shuffle was drawn. Its null SD/mean ratio ranges 0.16–1.98 and, in rep 1,
the 500 "permutations" cluster tightly around one value: the signature of a null carrying ~1 degree
of freedom rather than 500. Under the fix, the null mean is reproducible to 2% (0.00962 / 0.00968 /
0.00984) and the SD/mean ratio is a stable ~0.85, i.e. a properly dispersed max-cluster null.

Note also that the fix is **not uniformly "more" or "less" significant**: here it is more
conservative (1 surviving cluster vs 2–7), but which direction any given run moves depends on where
that run's single lucky/unlucky shuffle happened to land. Two of the three OLD reps on this exact
data would have reported 6–7 clusters.

Reproduce: `validate_permfix_ab.py` (it monkey-patches the pre-fix implementation back in for the
"old" arm), archived with its console log and results JSON in the consumer repo at
`analysis_results/paper1_analysis/spm/PERMFIX_2026-07-26/`.

---

## 3. The fix

```python
def perform_permutations(self):
    participants = np.arange(self.n_participants)
    if self.n_possible_unique_permutations <= self.n_permutations:
        # Exhaustive: every distinct ordering exactly once (identity first).
        self.permutation_matrix = np.array(list(itertools.permutations(participants)), dtype=int)
    else:
        n_perms = self.n_permutations
        self.permutation_matrix = np.zeros((n_perms, self.n_participants), dtype=int)
        self.permutation_matrix[0, :] = participants  # identity => observed statistic
        for iter_ in range(1, n_perms):
            self.permutation_matrix[iter_, :] = np.random.permutation(participants)
    self.n_permutations = self.permutation_matrix.shape[0]
```

Design points:

- **Row 0 stays the identity** so the observed statistic is part of its own null (Phipson & Smyth
  2010), giving the correct `p >= 1/n_perms` floor instead of a p-value that can reach 0.
- **The small-n exhaustive guard is preserved and is now genuinely exhaustive.** The old code took
  `n! - 1` *consecutive* items from an enumeration that began at a shuffled point and prepended the
  identity, so it could both duplicate the identity and omit one ordering. The new branch enumerates
  all `n!` orderings exactly once. It triggers for `n <= 7` at `n_permutations = 10,000`
  (`7! = 5,040 <= 10,000 < 8! = 40,320`).
- **Sampling with replacement in the random branch is intentional and standard.** Duplicate draws are
  expected (e.g. ~132 duplicate rows in 10,000 draws at n=9, birthday-paradox) and do not bias the
  null; forcing uniqueness would.
- `np.random` (global state) is used, matching `PermutationGroupLabels`. Runs are not seeded, so
  thresholds still carry Monte-Carlo noise — but now that noise is O(1/sqrt(n_perms)), not O(1).

---

## 4. Validation

### 4.1 Permutation-matrix diagnostics (post-fix)

```
n=   9  n_perms=10000  positions constant: 0/9    unique values/position: 9/9      dup rows=132  all rows valid perms=True  row0=identity
n= 222  n_perms=10000  positions constant: 0/222  unique values/position: 222/222  dup rows=0    all rows valid perms=True  row0=identity
n= 694  n_perms=10000  positions constant: 0/694  unique values/position: 694/694  dup rows=0    all rows valid perms=True  row0=identity

exhaustive branch:  n=3 -> 6 rows / 6 unique;  n=5 -> 120/120;  n=7 -> 5040/5040 (all identity-first)
random branch:      n=8 (8! = 40,320 > 10,000) -> 10,000 rows

uniformity: position of participant 0 over 10,000 draws at n=694 -> mean 350.4, SD 201.4
            (uniform expectation: mean 346.5, SD 200.3)
```

### 4.2 Test suite

`pytest` — **112 passed** (was 103; nine new regression tests added in
`tests/test_permutation_test.py::TestPermutationCorrelationTTestNonDegenerate`):

| test | guards against |
|---|---|
| `test_no_position_is_frozen_across_draws` | the exact failure mode (frozen head) |
| `test_every_position_explores_most_participants` | the "max 9 distinct values" cap |
| `test_participant_position_is_uniform` | a biased sampler |
| `test_identity_row_first` | losing the `p >= 1/n_perms` floor |
| `test_every_row_is_a_valid_permutation` | malformed index rows |
| `test_exhaustive_enumeration_is_complete_and_unique` (n=3,5,6) | duplicate/missing orderings in the small-n branch |
| `test_random_branch_used_when_factorial_exceeds_request` | the branch boundary at n=8 |

---

## 5. What downstream users must do

Any correlation or scalar-regression result produced before 2026-07-26 must be **re-run**, not
patched: the affected quantities are the cluster-extent threshold and every cluster p-value. The
per-vertex statistic maps (`stat_value`, `correlation_r`, `slope_per_unit`) are computed outside the
permutation loop and are unaffected — only the *inference* about which clusters survive changes.

ANOVA and two-sample t-test outputs need no action.
