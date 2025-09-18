import numpy as np
from bloch_vec_tomography import simulate_tomography, plot_results

n_runs = 1000
n_shots = 100
seed = 90

np.random.seed(seed)
r = np.random.randn(3)
r_true, r_estimates, F = simulate_tomography(r=r,
                                             n_runs=n_runs,
                                             n_shots=n_shots,
                                             seed=seed)
plot_results(r_true, r_estimates, F)