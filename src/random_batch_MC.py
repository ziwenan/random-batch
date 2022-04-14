# src/random_batch_MC.py
import numpy as np
from tqdm import tqdm


def random_batch_MC(x: np.ndarray, tau: float, T: float, U_long, U_short, V, w: float = 1, m: int = 9, beta: float = 1,
                    thinning: int = None) -> np.ndarray:
    '''
    This function operates the random batch Monte Carlo (RBMC) to sample from the equilibrium distribution for a
    many-body interacting particle system (IPS). External forces (denoted as V) and pairwise interactions (denoted as U)
    are considered in the IPS. Without loss of generality, a splitting strategy is implemented to U, i.e.,
    :math:`U(x) = U_1(x) + U_2(x)`, where :math:`U_1(x)` and :math:`U_2(x)` represent the long range and short range
    effects, respectively. For a thorough explanation, please find reference [1].

    Args:
        x: A one or two-dimensional numpy data array. If the particle system is multidimensional, the rows and columns
           of x correspond to particles and axes. If the data is one-dimensional, x is flattened.
        tau: Time interval over which RBM operates
        T: Time span
        U_long: A function that calculates the derivative of the interacting force (long range) between two particles in the system.
                The U_long function shall take (x_i - x_j) as the input and a float number representing the derivative
                of the outgoing force as the output. The function shall be symmetric over zero. E.g., lambda xixj: 1/xixj
        U_short: A function that calculates the interacting force (short range, not derivative) between two particles in the system.
                The U_short function shall take (x_i - x_j) as the input and a float number representing the outgoing
                force as the output. The function shall be symmetric over zero. E.g., lambda xixj: 1/xixj
        V: A function that calculates the derivative of external force. Input is the current location of a particle.
           Output is a float number. E.g., lambda x: x.
        w: Weight used to balance V and U. *extend to array*
        m: The number of moving steps. *rewrite*
        beta: Constant of the Gibbs distribution, positive
        thinning: Thinning of the Monte Carlo samples.

    Returns:
        A two or three-dimensional numpy array (time by particles or time by particles by axes) consisting the locations
        and time stamps for the particles.

    [1] Li, L., Xu, Z., & Zhao, Y. (2020). A random-batch Monte Carlo method for many-body systems with singular kernels.
        SIAM Journal on Scientific Computing, 42(3), A1486-A1509.
    '''

    # input error handling
    if x.ndim == 1:
        N = x.size
        d = 1
    elif x.ndim == 2:
        N = x.shape[0]
        d = x.shape[1]
    else:
        print('data must be 1 or 2-dimensional')
        return None

    if tau <= 0:
        print('tau must be greater than 0')
        raise ValueError

    if T // tau < 10:
        print('T must be at least 10 times larger than tau')
        raise ValueError

    if thinning is None:
        thinning = 1

    Nt = int(np.ceil(T / tau))

    x_curr = x.copy()
    count = 1
    # out = x[np.newaxis, ...]
    if d == 1:
        out = np.zeros((Nt // thinning + 1, N))
        out[0] = x
    else:
        out = np.zeros((Nt // thinning + 1, N, d))
        out[0] = x

    # main loop: random batch method - 1
    for _ in tqdm(range(Nt)):
        # choose a particle at random
        i = np.random.randint(0, N)
        x_prop = x_curr[i]

        # Step 1: moving step: move the particle with K1, long range effect, loop m times
        for k in range(m):
            # choose a second particle to interact with i at random
            j = np.random.randint(0, N - 1)
            j += j >= i
            K = U_long(x_prop - x_curr[j])
            x_prop -= tau * (V(x_curr[i]) / w / (N - 1) + K) + np.sqrt(
                2 * tau / beta / w / w / (N - 1)) * np.random.randn()

        # Step 2: accept-rejection step: compute the acceptance ratio with K2, short range effect, singular
        indices_except_i = [*range(N)]
        indices_except_i.remove(i)

        # compute the summation of U_short w.r.t. the proposed x_i
        # use vectorised function U_short if possible
        try:
            U_prop = sum(U_short(x_prop - x_curr[indices_except_i]))
        except ValueError:
            U_prop = sum(np.vectorize(U_short)(x_prop - x_curr[indices_except_i]))

        # compute summation of U_short w.r.t. the current x_i
        # use vectorised function U_short if possible
        try:
            U_curr = sum(U_short(x_curr[i] - x_curr[indices_except_i]))
        except ValueError:
            U_curr = sum(np.vectorize(U_short)(x_curr[i] - x_curr[indices_except_i]))

        alpha = np.exp(-beta * w * w * (U_prop - U_curr))
        if np.random.rand() < min(1, alpha):  # accept
            x_curr[i] = x_prop

        # save to the output array
        if (_ + 1) % thinning == 0:
            # out = np.append(out, x_curr[np.newaxis, ...], axis=0)
            out[count] = x_curr
            count += 1

    return out
