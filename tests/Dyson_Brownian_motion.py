# Dyson_Brownian_motion.py
import numpy as np
from src import random_batch_MC
from src import plot


# sample with random batch Monte Carlo
N = 100  # the number of particles
x = np.random.rand(N) * 10 - 5  # (random) different opinions at t = 0
tau = 1e-3 # time interval
T = 30  # time span
w = 1 / (N-1)
beta = (N-1) ** 2
m = 9
thinning = 100
V = lambda x: x
def U_long(x):
    r = np.abs(x)
    if r < 0.01:
        return -100*np.sign(x)
    else:
        return -np.sign(x)/r
def U_short(x):
    x = np.abs(x)
    return np.where(x >= 0.01, 0, -5.605170186 - np.log(x) + 100*x)
res = random_batch_MC(x, tau, T, U_long, U_short, V, w, m, beta, thinning)


plot.plot(res)

res_aug = np.stack([res, np.zeros(res.shape)], axis=2)
plot.animate(res_aug, scatter_kwargs={'s':1}, animate_kwargs={'interval':10})
