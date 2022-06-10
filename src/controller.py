# src/controller.py
import inspect
import sys
import warnings
from typing import Union

import numpy as np

from src.utils import wrapper_func


class Controller:
    '''
    **This is a subclass of IPS, not intended for use as a stand-alone component.**

    Controller is designed to modify locations/velocities of a set of particles.The target of modification,
    locations/velocities, is determined by the order (first/second) of an IPS.

    A controller either coerces the target to a fixed value, or uses a callable function to modify it. A controller
    object can be initialised with the "Controller.init" method. The advantage of using "Controller.init" over the
    default "__init__" method is that "Controller.init" performs an automatic validity check of the "func_or_value"
    argument on a giving testing array. The following three types of "func_or_value" argument are acceptable,
    (1) a non-vectorised function (operates on one particle per execution)
    (2) a vectorised function (operates on a set of particles per execution)
    (3) (a) fixed value(s) (scalar or array-like)

    Attributes:
        what (str): "function" or "value".
        function (Callable[..., Any]): A (possibly optimised) function.
        value (Union[int, float, complex, np.ndarray]): A scalar or np.ndarray.

    Examples:
        >>> def fn(y): # defines the operator (linear force that attracts the particles to the origin)
        >>>     return -y

        >>> x = np.random.rand(100, 2) # 100 random particles in a 2D space
        >>> V = Controller.init(fn, x) # initialise the controller with function fn
        >>> print(V)
        >>> V.apply_to(x)
    '''

    # TODO: cutoff
    def __init__(self, what=None, function=None, value=None, is_vectorised=False, use_numba=False):
        '''
        Do not use this. Use "Controller.init()" instead.

        Args:
            what (str): "function" or "value".
            function (Callable[..., Any]): A (possibly optimised) function.
            value (Union[int, float, complex, np.ndarray]): A scalar or np.ndarray.
            is_vectorised: Boolean, whether the function is vectorised.
            use_numba: Boolean, whether the function is optimised with numba just-in-time compiler.
        '''
        self.what = what
        self.function = function
        self.value = value
        self._is_vectorised = is_vectorised
        self._use_numba = use_numba

    @classmethod
    def init(cls, func_or_value, x, use_numba=True, force_nonvectorise=False, verbose=False, suppress_warnings=False,
             **kwargs):
        '''
        Recommended way to initialise a controller.

        A validity check will be performed with 'Controller.validity_check' method. The controller will be
        reworked/optimised (if needed) so that it always applies to a set of particles per execution.

        Args:
            func_or_value: The following three types of "func_or_value" argument are acceptable,
                          (1) a non-vectorised function (operates on one particle per execution),
                          (2) a vectorised function (operates on a set of particles per execution),
                          (3) (a) fixed value(s) (scalar or array-like).
                          If "func_or_value" is a callable function and the number of arguments of the function is more
                          than two, arguments must be correctly named. Name of the first argument must always be "x",
                          representing the target (positions/velocities or binary interactions of them in case of
                          interacting force) of the calculation.
            x: 1D (particles) or 2D (particles by axes) array of positions/velocities, for testing only.
            use_numba (bool): Only used when "func_or_value" is a function, use numba just-in-time compilation to
                              optimise code, works only if function takes one argument.
            force_nonvectorise (bool): Only used when "func_or_value" is a function, force non-vectorised mode if True.
                                       Prevent misuse of automatic vectorisation.
            verbose (bool): Verbose messages.
            suppress_warnings (bool): Suppress warning messages.
            kwargs: Other keyword arguments to pass into function.

        Returns:
            Controller.

        Raises:
            Input error: Fails to produce (a conformable) output array.
        '''
        # try:
        what, func, value, vectorised, numba_ = Controller.check_validity(func_or_value, x, use_numba,
                                                                          force_nonvectorise, verbose,
                                                                          suppress_warnings, **kwargs)
        return cls(what, func, value, vectorised, numba_)
        # except Exception as e:
        #     print(e)

    def apply_to(self, x, **kwargs):
        '''
        Apply controller to target x.

        Args:
            x: 1D (particles) or 2D (particles by axes) array of positions/velocities.
            kwargs: Other keyword arguments to pass into function.

        Returns:
            Numpy array, the modified property.
        '''
        if self.what == 'value':
            return self.value
        else:
            return self.function(x, **kwargs)

    @property
    def is_vectorised(self):
        '''Whether the function is vectorised.'''
        return self._is_vectorised

    @property
    def use_numba(self):
        '''Whether numba just-in-time compiler is used.'''
        return self._use_numba

    def __str__(self, args=''):
        if self.what == 'value':
            return f'fixed value: {self.value[0]}'
        else:
            return f'{str(self.function.__name__)} function({args}): vectorised: {self._is_vectorised}, numba: {self._use_numba}'

    def __repr__(self, args=''):
        if self.what == 'value':
            return f'fixed value: {self.value[0]}'
        else:
            return f'{str(self.function.__name__)} function({args}): vectorised: {self._is_vectorised}, numba: {self._use_numba}'

    def __call__(self, x, **kwargs):
        '''Same as Controller.apply_to()'''
        return self.apply_to(x, **kwargs)

    @staticmethod
    def check_validity(func_or_value, x, use_numba=True, force_nonvectorise=False, verbose=False,
                       suppress_warnings=False, **kwargs):
        '''
        Performs validity check of a given operator on a testing array x.

        Note that the function does not perform formal check of the calculation process. It runs the operator on the
        testing array, checks the shape of the returned value, and formats the returned array if needed. A reworked
        operator will be returned such that it always operates on an array ("fake vectorisation"). If the operator is a
        function, it will first be tested in vectorised mode and then non-vectorised mode.

        Args:
            func_or_value: The following three types of "func_or_value" argument are acceptable,
                          (1) a non-vectorised function (operates on one particle per execution),
                          (2) a vectorised function (operates on a set of particles per execution),
                          (3) (a) fixed value(s) (scalar or array-like).
                          If "func_or_value" is a callable function and the number of arguments of the function is more
                          than two, arguments must be correctly named. Name of the first argument must always be "x",
                          representing the target (positions/velocities or binary interactions of them in case of
                          interacting force) of the calculation.
            x: 1D (particles) or 2D (particles by axes) array of positions/velocities, for testing only.
            use_numba (bool): Only used when "func_or_value" is a function, use numba just-in-time compilation to
                              optimise code, works only if function takes one argument.
            force_nonvectorise (bool): Only used when "func_or_value" is a function, force non-vectorised mode if True.
                                       Prevent misuse of automatic vectorisation.
            verbose (bool):  Verbose messages.
            suppress_warnings (bool): Suppress warning messages.
            kwargs (dict): Other keyword arguments of the function if "func_or_value" is a callable function.

        Returns: Tuple (what, func, value, vectorised, numba_).
        '''

        x = np.asarray(x)
        dtype = x.dtype
        if x.ndim == 1:
            num = x.size
            d = 1
        elif x.ndim == 2:
            num = x.shape[0]
            d = x.shape[1]
        else:
            raise ValueError('"x" array must be 1D (particles) or 2D (particles by axes)')

        vectorised = numba_ = False
        func = value = None
        if callable(func_or_value):
            what = 'function'
        elif isinstance(func_or_value, (int, float, complex)):
            what = 'value'
            if d == 1:
                value = np.full(num, func_or_value)
            else:
                value = np.full((num, d), func_or_value)
        else:
            try:
                func_or_value = np.asarray(func_or_value)
            except:
                raise ValueError(
                    '"func_or_value" must be a callable function or a scalar value or a numpy array-like object') from None
            assert func_or_value.shape == (num, d) or func_or_value.shape == (
                d,), f'Non-conformable array. Shape of array-like object must be ({d},) or ({num}, {d})'
            what = 'value'
            value = np.full((num, d), func_or_value)
            # err_str = ("Controller must be one of the following: "
            #            "(1) a fixed number (int or float) or a (numpy) array of numbers, "
            #            "(2) a non-vectorised function that takes {0} as input and a float as output, "
            #            "(3) a vectorised function that returns a (numpy) array of the same shape as the input.")

        if what == 'function':
            fn_parameters = inspect.signature(func_or_value).parameters
            if len(fn_parameters.keys()) == 1:
                args = tuple([x])
                kwargs_all = dict()
            else:
                if 'x' not in fn_parameters.keys():
                    raise KeyError(f'"x" must be given as a keyword argument in function {func_or_value.__name__}')
                keys_diff = set(fn_parameters.keys()) - set(kwargs.keys()).union('x')
                if len(keys_diff) != 0:
                    args = ', '.join(str(x) for x in keys_diff)
                    raise KeyError(f'Missing argument(s) {args} in function {func_or_value.__name__}')
                args = tuple()
                kwargs_all = kwargs.copy()
                kwargs_all.update(x=x)
            if verbose:
                print(f'fn_parameters: {fn_parameters}')
                print(f'args: {args}')
                print(f'kwargs: {kwargs}')

            try:  # vectorised implementation
                if verbose:
                    print('vectorised computation...', end='')
                assert not force_nonvectorise
                if suppress_warnings:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Creating an ndarray from ragged nested sequences")
                        ret = np.asarray(func_or_value(*args, **kwargs_all))
                else:
                    ret = np.asarray(func_or_value(*args, **kwargs_all))
                if d == 1:
                    assert ret.size == num, f'Expected: ({num}, ), actual: {ret.size}'
                else:
                    assert ret.shape == (num, d), f'Expected: ({num}, {d}), actual: {ret.shape}'
                if ret.dtype == dtype:
                    func = func_or_value
                else:
                    def func(x, **kwargs):
                        return np.array(func_or_value(x, **kwargs), dtype=dtype)
                vectorised = True
                if verbose:
                    print('succeed', end='')
            except:  # fail for any reason
                if verbose:
                    print('\nserial computation...', end='')
                try:  # try non-vectorised implementation and rework the function
                    ret = np.asarray(func_or_value(x[0], **kwargs))
                    assert (d == 1 and ret.size == 1) or ret.shape == (d,), f'Expected: ({d},), actual: {ret.size}'
                    dtype_func_wrapper = None if ret.dtype == dtype else dtype
                    if use_numba:  # try to use numba
                        if kwargs:
                            if not suppress_warnings:
                                warnings.warn(
                                    f'Fall back to non-numba mode. Numba jit does not support **kwargs {kwargs}. Please define numba function manually.')
                            numba_ = False
                        else:
                            try:
                                if verbose:
                                    print('numba...', end='')
                                func_reworked = wrapper_func(func_or_value, x, d, use_numba, dtype_func_wrapper,
                                                             **kwargs)
                                func_reworked(x, **kwargs)
                                numba_ = True
                                if verbose:
                                    print('succeed')
                            except Exception as exc:
                                if not suppress_warnings:
                                    warnings.warn(
                                        f'Fall back to non-numba mode. Fail to apply numba.jit(nopython=True) to function {func_or_value.__name__}: ' + str(
                                            exc))
                                numba_ = False

                    if not numba_:  # not using numba
                        if verbose:
                            print('without numba...', end='')
                        func_reworked = wrapper_func(func_or_value, x, d, False, dtype_func_wrapper, **kwargs)
                        func_reworked(x, **kwargs)
                        if verbose:
                            print('succeed')

                except AssertionError as exc:
                    raise type(exc)(
                        f'Non-conformable output array from function {func_or_value.__name__}. ' + str(
                            exc)).with_traceback(sys.exc_info()[2]) from None
                except Exception as exc:
                    raise type(exc)(
                        f'The following error occurred while applying function {func_or_value.__name__} to data array: ' + str(
                            exc)).with_traceback(sys.exc_info()[2]) from None

                func = func_reworked

        return what, func, value, vectorised, numba_


if __name__ == '__main__':
    num_particles = 10
    # x_2d = np.random.rand(num_particles, 2)
    what, func, value, vectorised, numba_ = Controller.check_validity(
        lambda y: np.zeros((num_particles,)), np.ones([10, ]))
    what, func, value, vectorised, numba_ = Controller.check_validity(
        func_or_value=lambda y: np.zeros((num_particles, 2)), x=np.ones([10, 2]), verbose=True)
