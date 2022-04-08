# Random Batch
Random batch methods (RBM) for interacting particle system (IPS)


## Brief Introduction to IPS and RBM
### Binary interacting particle system

IPS is a stochastic process that describes the evolution of a population over time, e.g., Vicsek model for collective 
motion and swarm behaviour, flocking in school of fishes, group of birds, chemotaxis of bacteria, consensus clusters in 
opinion dynamics. The project focuses on IPS with binary interactions. The stochastic process for an N-particle IPS with
binary interactions is given by

<center><img src="https://render.githubusercontent.com/render/math?math=\color{blue}{dX^i=-\nabla V(X^i)dt %2b \dfrac{1}{N-1}\sum_{j:j \neq i} K(X^i-X^j)dt %2b \sigma dB^i,\ i=1,\ldots,N},"></center>

 
where <img src="https://render.githubusercontent.com/render/math?math=X^i,\ i=1,\ldots,N"> is the particle, V represents
the external force, K is the binary interacting kernel, B is the Brownian motion.

### Random batch methods

While direct computation of the fully coupled system could be prohibitively expensive when N is medium or large, RBM is
able to reduce the computational cost significantly from O(N^2) to O(N) per time step. The intuition behind RBM is to 
evolve the IPS in small batches of particles, while the law of large number guarantees asymptotic convergence as 
<img src="https://render.githubusercontent.com/render/math?math=N \rightarrow \infty"> under mild conditions. Two RBM methods are provided in this 
python project, namely RBM-1 and RBM-replace. RBM-1 shuffles and divides the particles into small batches per time step.
RBM-replace samples random particles with replacement. For more information, please resort to [1] and [2]. The pseudo 
code of algorithms RBM-1 and RBM-replace are given below (figure from [1]).

![RBM-1](fig/RBM-1.jpg?raw=true)
![RBM-r](fig/RBM-r.jpg?raw=true)

### Random batch Monte Carlo

Random batch Monte Carlo (RBMC) is a fast Markov chain Monte Carlo method to sample from the equilibrium distribution 
for a many-body IPS. The equilibrium state is an N-particle Gibbs distribution (or Boltzmann distribution). Suppose the 
pairwise interacting kernel between two particles is singular and long-tailed. RBMC features a splitting strategy, where 
the interacting kernel can be split into two separate parts, i.e., 

<center><img src="https://render.githubusercontent.com/render/math?math=\color{blue} K(X^i-X^j) = K_1(X^i-X^j) %2b K_2(X^i-X^j),"></center>

where <img src="https://render.githubusercontent.com/render/math?math=K_1(\cdot)"> and <img src="https://render.githubusercontent.com/render/math?math=K_2(\cdot)"> 
represent the long range and short range effects, respectively. For each iteration of the RBMC algorithm, a moving step 
and an accept-rejection step are carried out in turn. During the moving step, a random chosen particle X^i is updated by 
solving the stochastic differential equation with respect to the long range effect. The idea of dividing N particles 
into random mini-batches is implemented in the moving step to ease computational complexity.
In the accept-rejection step, short range effect is considered in the calculation of Metropolis acceptance ratio to 
guarantee convergence to the invariant distribution. For a thorough explanation, please find Algorithm 1 on page 10 of [3].

## Usages
* random_batch
* random_batch_replace
* random_batch_MC

---
## References
1. Jin, S., Li, L., & Liu, J. G. (2020). Random batch methods (RBM) for interacting particle systems. Journal of Computational Physics, 400, 108877.
2. Jin, S., & Li, L. (2022). Random batch methods for classical and quantum interacting particle systems and statistical samplings. In Active Particles, Volume 3 (pp. 153-200). Birkhäuser, Cham.
3. Li, L., Xu, Z., & Zhao, Y. (2020). A random-batch Monte Carlo method for many-body systems with singular kernels. SIAM Journal on Scientific Computing, 42(3), A1486-A1509.