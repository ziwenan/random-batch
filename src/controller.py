# src/controller.py
import numpy as np
# from numba import jit
from src.exceptions import InputError
from src.particles import Particles


class Controller:
    '''
    A controller is an operation to be applied on a single particle (non-vectorised operation) or a group of particles (vectorised operation).
    Initialise a controller with the class method 'Controller.init' will automatically check the validity of an operation.
    Input of a controller function should be the position of a single particle or a group of particles. Output must have the same dimension as the input.
    Three types of controller are supported:
    (1) non-vectorised function
    (2) vectorised function
    (3) scalar or array-like
    '''

    def __init__(self, operator, is_fixed=False, is_vectorised=False):
        self.operator = operator
        self._is_fixed = is_fixed
        self._is_vectorised = is_vectorised

    @classmethod
    def init(cls, operator, x):
        '''
        Recommended way to initialise a controller. Apply vectorisation if possible.
        Args:
            operator: Operator, one of the following three, vectorised function, non-vectorised function, scalar or array-like object.
            x: Array-like, used to check the validity of the given operator.

        Returns:
            A list of three components, a valid operator, a boolean (True if operator is fixed (scalar or array)), a boolean (True if operator is a vectorised function).
        '''
        try:
            valid_operator = Controller.is_valid(operator, x)
            return cls(valid_operator[0], valid_operator[1], valid_operator[2])
        except Exception as e:
            print(e)

    def apply(self, x):
        '''
        Apply controller to an array x.
        Args:
            x: Array-like (1-D or 2-D (num by dim)), data to apply the controller.

        Returns:
            Output numpy array.
        '''
        if self._is_fixed:
            return self.operator
        else:
            return self.operator(x)

    @property
    def is_fixed(self):
        return self._is_fixed

    @property
    def is_vectorised(self):
        return self._is_vectorised

    def __repr__(self):
        return f'operator {self.operator.__name__ if not self._is_fixed else ""}, fixed: {self._is_fixed}, vectorised: {self._is_vectorised}'

    def __str__(self):
        if self._is_vectorised:
            str = 'vectorised function'
        elif self._is_fixed:
            str = 'fixed'
        else:
            str = 'non-vectorised function'
        return f'{str} {self.operator.__name__ if not self._is_fixed else "value"}'

    @staticmethod
    def is_valid(operator, x):
        particles = Particles(x)
        if callable(operator):  # operator is a function
            is_fixed = is_vectorised = False

            try:  # vectorised computation
                ret = operator(particles.position)
                if particles.d != 1:
                    if np.asarray(ret).shape == (particles.num, particles.d):
                        is_vectorised = True
                        operator_valid = operator
                elif np.asarray(ret).size == particles.num:
                    is_vectorised = True
                    operator_valid = operator
            except:
                try:
                    ret = operator(particles.position[0])
                    if particles.d != 1:
                        assert np.asarray(ret).shape == (particles.d,)

                        # @jit(nopython=True)
                        # def operator_valid(array):
                        #     ret = []
                        #     for item in array:
                        #         ret.append(operator(item))
                        #     return ret
                        def operator_valid(array):
                            return np.stack([*map(operator, array)], axis=0)
                    else:
                        assert np.asarray(ret).size == 1

                        # @jit(nopython=True)
                        # def operator_valid(array):
                        #     ret = []
                        #     for item in array:
                        #         ret.append(operator(item))
                        #     return ret
                        def operator_valid(array):
                            return np.fromiter(map(operator, array), dtype=float)
                except AssertionError:
                    raise InputError(operator.__name__, 'Non-conformable output array.') from None
                except:
                    raise InputError(operator.__name__, 'Fail to apply function to data array.') from None
            else:
                if not is_vectorised:
                    try:
                        ret = operator(particles.position[0])
                        if particles.d != 1:
                            assert np.asarray(ret).shape == (particles.d,)

                            # @jit(nopython=True)
                            # def operator_valid(array):
                            #     ret = []
                            #     for item in array:
                            #         ret.append(operator(item))
                            #     return ret
                            def operator_valid(array):
                                return np.stack([*map(operator, array)], axis=0)
                        else:
                            assert np.asarray(ret).size == 1

                            # @jit(nopython=True)
                            # def operator_valid(array):
                            #     ret = []
                            #     for item in array:
                            #         ret.append(operator(item))
                            #     return ret
                            def operator_valid(array):
                                return np.fromiter(map(operator, array), dtype=float)
                    except AssertionError:
                        raise InputError(operator.__name__, 'Non-conformable output array.') from None
                    except:
                        raise InputError(operator.__name__, 'Fail to apply function to data array.') from None
        elif isinstance(operator, (int, float)):  # operator is a scalar
            is_fixed, is_vectorised = True, False
            if particles.d == 1:
                operator_valid = np.full(particles.num, operator)
            else:
                operator_valid = np.full((particles.num, particles.d), operator)
        elif isinstance(operator, (list, np.ndarray)) and particles.d != 1:  # operator is array-like
            is_fixed, is_vectorised = True, False
            assert np.asarray(operator).shape == (particles.d,), 'Non-conformable array.'
            operator_valid = np.full((particles.num, particles.d), np.asarray(operator))
        else:
            err_str = ("Controller must be one of the following: "
                       "(1) a fixed number (int or float) or a (numpy) array of numbers, "
                       "(2) a non-vectorised function that takes {0} as input and a float as output, "
                       "(3) a vectorised function that returns a (numpy) array of the same shape as the input.")
            raise ValueError(err_str)
        return [operator_valid, is_fixed, is_vectorised]


if __name__ == '__main__':
    def fn(y):
        return -y

    x = np.random.rand(100)
    V = Controller.init(fn, x)
    print(V)
    V.apply(x)
