import matplotlib.pyplot as plt
import numpy as np
from src import random_batch
from src import random_batch_replace


N = 100
x = np.random.rand(N) * 10
tau = 1e-3
T = 3
alpha = 40
kernel = lambda x: np.logical_and(np.abs(x) >= 0, np.abs(x) <= 1) * (-x) * alpha
V = lambda x: 0
sigma = 1e-4
res1 = random_batch(x, tau, T, kernel, V, sigma, num = 100)
res2 = random_batch_replace(x, tau, T, kernel, V, sigma, num = 100)


fig, axs = plt.subplots(1, 2)
for i in range(N):
    axs[0].plot(res1[:, i], color="blue",linewidth=.5)

for i in range(N):
    axs[1].plot(res2[:, i], color="blue",linewidth=.5)