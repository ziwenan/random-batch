import numpy as np
from tqdm import tqdm


def random_batch(x: np.ndarray, tau: float, T: float, kernel, V, sigma: float = 0, num: int = None) -> np.ndarray:
    '''
    This function operates the random batch method - 1 (RBM-1) for interacting particle system (IPS). RBM-1 shuffles and
    divides the particles into small batches, and evolves the IPS on batch basis.

    The method could significantly reduce the computational cost of an interacting particle system with binary
    interactions. The stochastic differential equation describing an N-particle binary IPS is given by

    .. math:: dX^i = -\\nabla V(X^i)dt + \\dfrac{1}{N-1} \\sum_{j:j \\neq i} K(X^i-X^j)dt + \\sigma dB^i, i=1,\\ldots,N.

    RBM solves the binary IPS by dividing the particles into small batches. The computational complexity of RBM is O(N)
    per time step. The batch capacity considered in this function is two.

    Args:
        x: A one or two-dimensional numpy data array. If the particle system is multidimensional, the rows and columns
           of x correspond to particles and axes. If the data is one-dimensional, x is flattened.
        tau: Time interval over which RBM operates
        T: Time span
        kernel: A function that calculates the interacting force between two particles in the system. The kernel
                function shall take (x_i - x_j) as the input and a float number representing the outgoing force as the
                output. The kernel function shall be symmetric over zero. E.g., lambda xixj: 1/xixj
        V: A function that calculates the external force, i.e., gradient of the external field. Input is the location.
           Output is a float number.
        sigma: Diffusion term
        num: Maximum number of time points to be returned. If None, num equals to ceil(T/tau)

    Returns:
        A two or three-dimensional numpy array (time by particles or time by particles by axes) consisting the locations
        and time stamps of the particles.
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

    Nt = int(np.ceil(T / tau))

    if num is None:
        num = Nt
    elif num <= 0:
        print('num must be greater than 0')
        raise ValueError
    else:
        num = min(num, Nt)

    indices = np.arange(N)
    next_out = 1
    x_curr = x.copy()
    if d == 1:
        out = np.zeros((num, N))
        out[0, :] = x
    else:
        out = np.zeros((num, N, d))
        out[0, :, :] = x
    m_out = np.linspace(0, Nt - 1, num)

    # main loop: random batch method - 1
    for m in tqdm(range(Nt)):
        # random divide N particles into n batches, the default batch capacity is 2
        np.random.shuffle(indices)

        x_prev = x_curr.copy()

        # loop over batches
        for q in range(N // 2):
            i = indices[q << 1]
            j = indices[(q << 1) + 1]
            K = kernel(x_prev[i] - x_prev[j])
            x_curr[i] += tau * (-V(x_prev[i]) + K) + sigma * np.sqrt(tau) * np.random.randn()
            x_curr[j] += tau * (-V(x_prev[j]) - K) + sigma * np.sqrt(tau) * np.random.randn()

        # save to the output array
        if m >= m_out[next_out]:
            if d == 1:
                out[next_out, :] = x_curr
            else:
                out[next_out, :, :] = x_curr
            next_out += 1

    return out
