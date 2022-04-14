# Stochastic opinion dynamics

This is the description file for python code "stochastic_opinion_dynamics.py".

## Model

Opinion dynamics is a self-organised dynamics, where the system is governed solely by interactions among individuals,
with the tendency to adjust to their "environment averages". This, in turn, leads to the formation of clusters. In the
stochastic opinion dynamics model, opinions of N agents interact with each other until reaching a consensus (if
possible). We use the random batch methods to solve the first-order opinion dynamics system given by

<center><img src="https://render.githubusercontent.com/render/math?math=\color{blue}{dX^i=\dfrac{\alpha}{N-1} \sum_{j \neq i} \phi(|X^j - X^i|)(X^j - X^i)dt %2b \sigma dB^i,\ i=1,\ldots,N},"></center>

where &alpha; is a scaling parameter, &phi;(&sdot;) is the influence function which acts on the “difference of
opinions” |x_i − x_j|. Since this is a self-organised system, V is set to a constant 0.


## Example 1 (1-D)
### Parameters
```python
N = 100  # the number of particles
x = np.random.rand(N) * 10  # (random) different opinions at t = 0
tau = 1e-3  # time interval
T = 3  # time span
alpha = 40  # scaling parameter
phi = lambda y: abs(y) <= 1 # influence function
kernel = lambda y: alpha * phi(y) * (-y) # binary interacting kernel
V = lambda x: 0  # external force, set to zero
sigma = N ** -1/3  # diffusion term of the Brownian motion
```

### Running and plotting
```python
fig, axs = plt.subplots(1, 2)
plot.plot(res1_rbm, ax=axs[0])
plot.plot(res1_rbmr, ax=axs[1])
```

![RBM-1](../fig/sod1.png?raw=true)

## Example 2 (2-D)
### Parameters
```python
N = 200  # the number of particles
x = np.random.rand(N, 2) * 10  # (random) different opinions at t = 0
tau = 1e-3  # time interval
T = 5  # time span
alpha = 10  # scaling parameter
phi = lambda y: abs(y) <= 1 # influence function
kernel = lambda y: alpha * phi(y) * (-y) # binary interacting kernel
V = lambda x: 0  # external force, set to zero
sigma = N ** -1/3  # diffusion term of the Brownian motion

```

### Running and plotting
```python
res2 = random_batch(x, tau, T, kernel, V, sigma, num=500)
plot.animate(res2, scatter_kwargs={'s':5}, animate_kwargs={'interval':10})
```

![RBM-1](../fig/sod2.gif?raw=true)


## References

1. Jin, S., Li, L., & Liu, J. G. (2020). Random batch methods (RBM) for interacting particle systems. Journal of
   Computational Physics, 400, 108877.
2. Motsch, S., & Tadmor, E. (2014). Heterophilious dynamics enhances consensus. SIAM review, 56(4), 577-621. 

 