# Random Batch
Random batch methods (RBM) for interacting particle system (IPS)

---

## Brief Intro to IPS and RBM
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
RBM-replace samples random particles with replacement. For more information, please resort to [1] and [2].
---

## Usages
* random_batch
* random_batch_replace
* random_batch_MC

---
## References
1. Jin, S., Li, L., & Liu, J. G. (2020). Random batch methods (RBM) for interacting particle systems. Journal of Computational Physics, 400, 108877.
2. Jin, S., & Li, L. (2022). Random batch methods for classical and quantum interacting particle systems and statistical samplings. In Active Particles, Volume 3 (pp. 153-200). Birkhäuser, Cham.