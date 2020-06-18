n_iterations = 10000
permutations = itertools.permutations(np.arange(rand_data.shape[0]))

rand_data_of_interest = rand_data[:, location_cartilage, :]

n_possible_permutations = factorial(rand_data_of_interest.shape[0])
if n_possible_permutations < n_iterations:
    n_iterations = n_possible_permutations

permutations_matrix = np.zeros((n_iterations,) + rand_data_of_interest.shape)
permutations_matrix[:, :, :, 0] = rand_data_of_interest[:, :, 0]

perms = np.zeros((n_iterations, rand_data_of_interest.shape[0]), dtype=np.int)
for iter_ in range(n_iterations):
    perms[iter_, :] = next(permutations)

for iter_ in range(n_iterations):
    permutations_matrix[iter_, :, :, 1] = rand_data_of_interest[perms[iter_, :], :, 1]
perms = None
