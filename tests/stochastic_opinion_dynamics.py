# stochastic_opinion_dynamics.py
import matplotlib.pyplot as plt
import numpy as np
from src import random_batch
from src import random_batch_replace
from src import plot


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
V = lambda x: 0  # external force, set to zero
sigma = N ** -1/3  # diffusion term of the Brownian motion

## run with RBM-1 and RBM-r
res1_rbm = random_batch(x, tau, T, kernel, V, sigma, num=100)
res1_rbmr = random_batch_replace(x, tau, T, kernel, V, sigma, num=100)

## plotting the results
fig, axs = plt.subplots(1, 2)
plot.plot(res1_rbm, ax=axs[0])
plot.plot(res1_rbmr, ax=axs[1])



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
V = lambda x: 0  # external force, set to zero
sigma = N ** -1/3  # diffusion term of the Brownian motion

## run with RBM-1
res2 = random_batch(x, tau, T, kernel, V, sigma, num=500)

## plotting the results
plot.animate(res2, scatter_kwargs={'s':5}, animate_kwargs={'interval':10})

