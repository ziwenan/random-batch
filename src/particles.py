# src/particles
import numpy as np


# import warnings


class Particle:
    '''
    Representation of a single particle. The dimension of the space is d. Visualisation is supported when d is at most three.
    '''

    __slots__ = ['_d', 'position', 'velocity', 'lower', 'upper', '_id']

    def __init__(self, position, velocity=None, lower=None, upper=None, id: int = None):
        assert position is not None, f'Position must not be None'
        self._d = 1 if np.isscalar(position) else np.size(position)
        self.position = self._set_value(position, 'position')
        self.velocity = self._set_value(velocity, 'velocity')
        self.lower = self._set_value(lower, 'lower')
        self.upper = self._set_value(upper, 'upper')
        self._id = 0 if id is None else id

    def _set_value(self, val, operator_name):
        if val is None:
            return 0 if self._d == 1 else np.zeros(self._d)
        else:
            val_dim = 1 if np.isscalar(val) else np.size(val)
            assert self._d == val_dim, f'The dimension of {operator_name}:{val_dim} should be {self._d}'
            return val if self._d == 1 else np.asarray(val)

    def init_graphic(self):
        pass

    def move(self):
        pass

    @property
    def d(self):
        return self._d

    @property
    def id(self):
        return self._id

    def __str__(self):
        return f'particle id{self._id}, location: {self.position}, velocity: {self.velocity}'


class Particles:
    '''
    Representation of a group of particles. The dimension of the space is d. Matplotlib visualisation is supported when d <= 3.
    '''

    __slots__ = ['_d', '_num', 'position', 'velocity', 'lower', 'upper', '_id', 'speed']

    def __init__(self, position, velocity=None, lower=None, upper=None, id=None):
        assert position is not None, f'Position must not be None'
        position = np.asarray(position)
        if position.ndim == 1:
            self._num = position.size
            self._d = 1
        elif position.ndim == 2:
            self._num = position.shape[0]
            self._d = position.shape[1]
        else:
            raise ValueError('"position" array must be 1-D or 2-D')
        self.position = self._set_value(position, 'position')
        self.velocity = self._set_value(velocity, 'velocity')
        self.calc_speed()

        if lower is not None and not np.isscalar(lower):
            lower = np.asarray(lower)
            assert lower.ndim == 1 and lower.size == self._d, f'"lower" must be an array-like object of size {self._d}'
        self.lower = lower

        if upper is not None and not np.isscalar(upper):
            upper = np.asarray(upper)
            assert upper.ndim == 1 and upper.size == self._d, f'"upper" must be an array-like object of size {self._d}'
        self.upper = upper
        self.apply_boundary_condition()
        self._id = np.arange(self._num) if id is None or len(id) != self._num else id

    def _set_value(self, val, operator_name):
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
                raise ValueError(f'The dimension of {operator_name}:{val.shape} should be ({self._num}, {self._d})')

    def apply_boundary_condition(self):
        if self.lower is not None:
            if self._d == 1:
                count = np.sum(self.position < self.lower)
                self.position = np.where(self.position < self.lower, self.lower, self.position)
                if count > 0:
                    print(f'Apply lower bound to {count} particles')
            else:
                count = 0
                for i in range(self._d):
                    count += np.sum(self.position[:, i] < self.lower[i])
                    self.position[:, i] = np.where(self.position[:, i] < self.lower[i], self.lower[i],
                                                   self.position[:, i])
                if count > 0:
                    print(f'Apply lower bound to {count} particles')
        if self.upper is not None:
            if self._d == 1:
                count = np.sum(self.position > self.upper)
                self.position = np.where(self.position > self.upper, self.upper, self.position)
                if count > 0:
                    print(f'Apply upper bound to {count} particles')
            else:
                count = 0
                for i in range(self._d):
                    count += np.sum(self.position[:, i] > self.upper[i])
                    self.position[:, i] = np.where(self.position[:, i] > self.upper[i], self.upper[i],
                                                   self.position[:, i])
                if count > 0:
                    print(f'Apply upper bound to {count} particles')

    def calc_speed(self):
        if self._d == 1:
            self.speed = np.abs(self.velocity)
        else:
            self.speed = np.sqrt(np.sum(self.velocity * self.velocity, axis=1))

    def shuffle(self):
        '''
        Shuffle particles.
        '''
        index = np.arange(self._num)
        np.random.shuffle(index)
        self.position = self.position[index]
        self.velocity = self.velocity[index]
        self.calc_speed()
        self._id = self._id[index]

    def pairwise_diff(self):
        '''
        Compute the pairwise position difference between particles. Every two consecutive particles are grouped as a pair, e.g., (0, 1), (2, 3), ..., (2*k, 2*k+1) are
        grouped together during computation.

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
        return self._d

    @property
    def num(self):
        return self._num

    @property
    def id(self):
        return self._id

    def __str__(self):
        return f'{self._num} particles in {self._d}-D space'


if __name__ == '__main__':
    par = Particles(np.arange(10).reshape([10]))
    print(par.position)
    print(par)
    print(par.pairwise_diff())

    par.shuffle()
    print(par.position)
    print(par.id)
