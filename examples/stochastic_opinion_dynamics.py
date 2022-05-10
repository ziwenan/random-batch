# stochastic_opinion_dynamics.py
import time
# import matplotlib.pyplot as plt
import numpy as np
from src.interacting_particle_system import IPS
from src import plot
from tqdm import tqdm


# This python script shows an application of random batch for the stochastic opinion dynamics example. See the
# associated markdown file for a description of the model.

############################
### 1-D Opinion Dynamics ###
############################

## setting up parameters
N = 100  # the number of particles
x = np.random.rand(N) * 10  # (random) different opinions at t = 0
tau = 1e-3  # time interval
T = 3  # time span
alpha = 40  # scaling parameter
phi = lambda y: abs(y) <= 1 # influence function
# phi = lambda y: 0.1 * (0<=np.abs(y)<=1) + 0.9 * (0.7071<=np.abs(y)<=1)
kernel = lambda y: alpha * phi(y) * (-y) # binary interacting kernel
V = 0  # external force, set to zero
sigma = N ** -1/3  # diffusion term of the Brownian motion

## run with RBM-1 and RBM-r
ips_rbm = IPS(x, V, kernel)
print(ips_rbm)
ips_generator = ips_rbm.evolve(tau, T)
res_rbm = np.stack(ips_generator, axis=0)
plot.plot(res_rbm)
# res1_rbm = random_batch(x, tau, T, kernel, V, sigma, num=100)
# res1_rbmr = random_batch_replace(x, tau, T, kernel, V, sigma, num=100)

## plotting the results
# fig, axs = plt.subplots(1, 2)
# plot.plot(res_rbm, ax=axs[0])
# plot.plot(res1_rbmr, ax=axs[1])



############################
### 2-D Opinion Dynamics ###
############################

## setting up parameters
N = 200  # the number of particles
x = np.random.rand(N, 2) * 10  # (random) different opinions at t = 0
tau = 1e-3  # time interval
T = 5  # time span
alpha = 10  # scaling parameter
phi = lambda y: abs(y) <= 1 # influence function
kernel = lambda y: alpha * phi(y) * (-y) # binary interacting kernel
V = 0  # external force, set to zero
sigma = N ** -1/3  # diffusion term of the Brownian motion

## run with RBM-1
ips2_rbm = IPS(x, V, kernel)
print(ips2_rbm)
ips_generator = ips2_rbm.evolve(tau, T, thinning=50)

t0 = time.time()
res2_rbm = np.stack(ips_generator, axis=0)
print(round(time.time() - t0, 2), 'sec')

# res = np.ndarray((5000, 200, 2))
# for i in tqdm(range(5000)):
#     res[i] = next(ips_generator)
#
# i = 0
# for item in tqdm(ips_generator, total=5000):
#     # res = np.append(res, item, axis=0)
#     res[i] = item
#     i += 1


## plotting the results
plot.animate(res2_rbm, scatter_kwoperators={'s':5}, animate_kwoperators={'interval':1})

