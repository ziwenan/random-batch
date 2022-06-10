# src/particles.py

from warnings import warn

import numpy as np


class Particles:
    '''
    Representation of a set of particles.

    There is no restriction to the dimension of the space (d). Position and velocity are stored as 2D numpy array when
    d>=2.

    Attributes:
        position: 1D (particles) or 2D (particles by axes) numpy array, position of the particles.
        velocity: 1D (particles) or 2D (particles by axes) numpy array, velocity of the particles.
        acceleration: 1D (particles) or 2D (particles by axes) numpy array, acceleration of the particles.
        lower: Lower bound of the particle space.
        upper: Upper bound of the particle space.
        boundary_condition (str): Boundary condition of the particle space.
        column_names (List(str)): List of column names.
        speed: Numpy array, speed of the particles.
    '''

    __slots__ = ['_d', '_num', 'position', 'velocity', 'acceleration', 'lower', 'upper', 'boundary_condition', '_index',
                 'column_names','speed']

    def __init__(self, position, velocity=None, acceleration=None, lower=None, upper=None, boundary_condition=None,
                 column_names=None, index=None):
        '''
        Initialise a particles class.

        Args:
            position: 1D (particles) or 2D (particles by axes) array-like object, position of the particles.
            velocity: 1D (particles) or 2D (particles by axes) array-like object, velocity of the particles.
            acceleration: 1D (particles) or 2D (particles by axes) array-like object, velocity of the particles.
            lower: Lower bound of the particle space, length should match the dimension of the space.
            upper: Upper bound of the particle space, length should match the dimension of the space.
            boundary_condition (str): Boundary condition of the particle space, None or "absorbing" or "elastic" or "periodic".
            column_names (List[str]): List of column names.
            index: Array, index of the particles.
        '''
        # TODO: rework absorbing boundary condition
        assert position is not None, f'Position must not be None'
        position = np.asarray(position)
        if position.ndim == 1:
            self._num = position.size
            self._d = 1
        elif position.ndim == 2:
            self._num = position.shape[0]
            self._d = position.shape[1]
        else:
            raise ValueError('"position" array must be 1D (particles) or 2D (particles by axes)')
        self.position = self._set_value(position, 'position')
        self.velocity = self._set_value(velocity, 'velocity')
        self.acceleration = self._set_value(acceleration, 'acceleration')
        self.calc_speed()

        if boundary_condition not in [None, 'absorbing', 'elastic', 'periodic']:
            raise ValueError(f'"boundary_condition" must be None or "absorbing" or "elastic" or "periodic".')
        if boundary_condition is not None and lower is None and upper is None:
            warn(f'"lower" and "upper" are not given for "boundary_condition" {boundary_condition}')
            boundary_condition = None
        if lower is not None:
            if self._d == 1:
                assert np.isscalar(lower)
            else:
                lower = np.asarray(lower)
                assert lower.ndim == 1 and lower.size == self._d, f'"lower" must be an array-like object of size {self._d}'
            if boundary_condition is None:
                boundary_condition = 'absorbing'
        if upper is not None:
            if self._d == 1:
                assert np.isscalar(upper)
            else:
                upper = np.asarray(upper)
                assert upper.ndim == 1 and upper.size == self._d, f'"upper" must be an array-like object of size {self._d}'
            if boundary_condition is None:
                boundary_condition = 'absorbing'
        self.lower = lower
        self.upper = upper
        self.boundary_condition = boundary_condition
        self.apply_boundary_condition()
        if column_names is None:
            if self._d == 1:
                self.column_names = None
            elif self._d == 2:
                self.column_names = ['x', 'y']
            elif self._d == 3:
                self.column_names = ['x', 'y', 'z']
            else:
                self.column_names = ['axis'+str(s) for s in range(self._d)]
        elif len(column_names) == self._d:
            self.column_names = [str(s) for s in column_names]
        else:
            raise ValueError(f'Invalid "column_names", {column_names}')
        if index is None:
            self._index = np.arange(self._num)
        else:
            uniq, index = np.unique(index, return_index=True)
            index = uniq[index.argsort()]
            if len(index) == self._num:
                self._index = index
            else:
                self._index = np.arange(self._num)

    def _set_value(self, val, name):
        if val is None:
            if self._d == 1:
                return np.zeros(self._num)
            else:
                return np.zeros([self._num, self._d])
        else:
            val = np.asarray(val)
            if self._d == 1 and val.ndim == 1:
                return val
            elif val.ndim == 2 and (self._num, self._d) == val.shape:
                return val
            else:
                raise ValueError(f'The dimension of {name}:{val.shape} should be ({self._num}, {self._d})')

    def apply_boundary_condition(self, verbose=False):
        '''
        Apply boundary condition to the particles.

        Readjust the position of particles according to "self.boundary_condition". If boundary condition is
        "absorbing", particles outside of boundaries are relocated to the closest boundary. If boundart condition is 
        "elastic" or "periodic", TODO.

        Args:
            verbose: Boolean, display messages when applying boundary condition to particles.
        '''
        if self.boundary_condition is not None:
            if self.lower is not None:
                if self._d == 1:
                    count = 0
                    if verbose:
                        count = np.sum(self.position < self.lower)
                    if self.boundary_condition == 'absorbing':
                        self.position = np.where(self.position < self.lower, self.lower, self.position)
                    elif self.boundary_condition == 'elastic':  # elastic and periodic
                        pass
                    if verbose and count > 0:
                        print(f'Apply lower bound to {count} particles')
                else:
                    count = 0
                    for i in range(self._d):
                        if verbose:
                            count += np.sum(self.position[:, i] < self.lower[i])
                        if self.boundary_condition == 'absorbing':
                            self.position[:, i] = np.where(self.position[:, i] < self.lower[i], self.lower[i],
                                                           self.position[:, i])
                        elif self.boundary_condition == 'elastic':
                            pass
                    if verbose and count > 0:
                        print(f'Apply lower bound to {count} particles')
            if self.upper is not None:
                if self._d == 1:
                    count = 0
                    if verbose:
                        count = np.sum(self.position > self.upper)
                    if self.boundary_condition == 'absorbing':
                        self.position = np.where(self.position > self.upper, self.upper, self.position)
                    elif self.boundary_condition == 'elastic':
                        pass
                    if verbose and count > 0:
                        print(f'Apply upper bound to {count} particles')
                else:
                    count = 0
                    for i in range(self._d):
                        if verbose:
                            count += np.sum(self.position[:, i] > self.upper[i])
                        if self.boundary_condition == 'absorbing':
                            self.position[:, i] = np.where(self.position[:, i] > self.upper[i], self.upper[i],
                                                           self.position[:, i])
                        elif self.boundary_condition == 'elastic':
                            pass
                    if verbose and count > 0:
                        print(f'Apply upper bound to {count} particles')

    def calc_speed(self):
        '''Calculate speed.'''
        if self._d == 1:
            self.speed = np.abs(self.velocity)
        else:
            self.speed = np.sqrt(np.sum(self.velocity * self.velocity, axis=1))

    def shuffle(self, seed=None, return_index=False):
        '''Shuffle particle positions, velocities, accelerations.'''
        index = np.arange(self._num)
        np.random.seed(seed)
        np.random.shuffle(index)
        self.position = self.position[index]
        self.velocity = self.velocity[index]
        self.acceleration = self.acceleration[index]
        self.calc_speed()
        self._index = self._index[index]
        if return_index:
            return index
        else:
            return None

    def pairwise_diff(self):
        '''
        Compute the pairwise position difference between two particles. Every two consecutive particles 
        are grouped as a pair, e.g., (0, 1), (2, 3), ..., (2*k, 2*k+1) are grouped together during computation.

        Returns:
            A numpy array of pairwise differences.
        '''
        pos_flip = self.position.copy()
        for i in range(0, self.num // 2 * 2, 2):
            pos_flip[[i, i + 1]] = pos_flip[[i + 1, i]]
        return self.position - pos_flip

    def init_graphic(self):
        pass

    @property
    def d(self):
        '''int: Dimension of the particle space.'''
        return self._d

    @property
    def num(self):
        '''int: Total number of the particles.'''
        return self._num

    @property
    def index(self):
        '''int[:]: Indices of the particles.'''
        return self._index

    def __str__(self):
        string = []
        string.append(f'{self._num} particles in {self._d}D space')
        string.append(None if self.boundary_condition is None else f'boundary: {self.boundary_condition}')
        return ', '.join(s for s in string if s)


if __name__ == '__main__':
    par = Particles(np.arange(10).reshape([10]))
    print(par.position)
    print(par)
    print(par.pairwise_diff())

    par.shuffle()
    print(par.position)
    print(par.id)
