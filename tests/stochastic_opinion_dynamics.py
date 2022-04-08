# stochastic_opinion_dynamics.py
import matplotlib.pyplot as plt
import numpy as np
from src import random_batch
from src import random_batch_replace

# This python script shows an application of random batch for the stochastic opinion dynamics example (see [1,
# 2] for detail). Opinion dynamics models the clustering behaviour of a self-organised system, where opinions of N
# agents interact with each other until reaching a consensus.
#
# References
# [1] Motsch, S., & Tadmor, E. (2014). Heterophilious dynamics enhances consensus. SIAM review, 56(4), 577-621.
# [2] Jin, S., Li, L., & Liu, J. G. (2020). Random batch methods (RBM) for interacting particle systems. Journal of
#     Computational Physics, 400, 108877.


## setting up parameters
N = 100  # the number of particles
x = np.random.rand(N) * 10  # (random) different opinions at t = 0
tau = 1e-3  # time interval
T = 3  # time span
alpha = 40  # scaling parameter
kernel = lambda x: np.logical_and(np.abs(x) >= 0, np.abs(x) <= 1) * (-x) * alpha  # binary interacting kernel
V = lambda x: 0  # external force, set to zero
sigma = 1e-4  # diffusion term of the Brownian motion

## run with RBM-1 and RBM-r
res1 = random_batch(x, tau, T, kernel, V, sigma, num=100)
res2 = random_batch_replace(x, tau, T, kernel, V, sigma, num=100)

## plotting the results
fig, axs = plt.subplots(1, 2)
for i in range(N):
    axs[0].plot(res1[:, i], color="blue", linewidth=.5)

for i in range(N):
    axs[1].plot(res2[:, i], color="blue", linewidth=.5)
