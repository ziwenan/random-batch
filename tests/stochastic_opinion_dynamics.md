# Stochastic opinion dynamics

Opinion dynamics is a self-organised dynamics, where the system is governed solely by interactions among individuals, 
with the tendency to adjust to their "environment averages". This, in turn, leads to the formation of clusters. In the 
stochastic opinion dynamics model, opinions of N agents interact with each other until reaching a consensus. The 
stochastic differential equation of a first-order opinion dynamics is given by

<center><img src="https://render.githubusercontent.com/render/math?math=\color{blue}{dX^i=\dfrac{\alpha}{N-1} \sum_{j \neq i} \phi(|X^j - X^i|)(X^j - X^i)dt %2b \sigma_N dB^i,\ i=1,\ldots,N},"></center>

where $\phi$ is a $`\sqrt{2}`$ 

```math
SE = \frac{\sigma}{\sqrt{n}}
```

# References
1. Motsch, S., & Tadmor, E. (2014). Heterophilious dynamics enhances consensus. SIAM review, 56(4), 577-621. 
2. Jin, S., Li, L., & Liu, J. G. (2020). Random batch methods (RBM) for interacting particle systems. Journal of
   Computational Physics, 400, 108877.
