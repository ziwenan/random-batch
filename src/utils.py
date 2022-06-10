# src/utils.py

import numpy as np


def wrapper_func(func, x, d, use_numba, dtype=None, list_comp=True, **kwargs):
    '''Decorator: rework func, use numba, coerce dtype.'''
    if use_numba and not kwargs:
        from numba import njit
        func_njit = njit()(func)

        @njit
        def func_reworked(arr):
            res = np.empty(x.shape)
            for i in range(x.shape[0]):
                res[i] = func_njit(arr[i])
            return res

        if dtype is None:
            inner = func_reworked
        else:
            def inner(arr):
                return np.array(func_reworked(arr), dtype=dtype)

        return inner
    else:
        if dtype is None:
            func_reworked = func
        else:
            def func_reworked(x, **kwargs):
                return np.array(func(x, **kwargs), dtype=dtype)

    # define inner func
    if kwargs:
        if d == 1:
            def inner(arr, **kwargs):
                if list_comp:
                    ret = np.array([func_reworked(y, **kwargs) for y in arr], dtype=dtype)  # list comprehension
                else:
                    ret = np.fromiter(map(lambda y: func_reworked(y, **kwargs), arr), dtype=dtype)  # map
                return ret
        else:
            def inner(arr, **kwargs):
                if list_comp:
                    ret = [func_reworked(y, **kwargs) for y in arr]  # list comprehension
                else:
                    ret = [*map(lambda y: func_reworked(y, **kwargs), arr)]  # map
                return np.stack(ret, axis=0)
    else:
        if d == 1:
            def inner(arr):
                if list_comp:
                    ret = np.array([func_reworked(y) for y in arr], dtype=dtype)  # list comprehension
                else:
                    ret = np.fromiter(map(func_reworked, arr), dtype=dtype)  # map
                return ret
        else:
            def inner(arr):
                if list_comp:
                    ret = [func_reworked(y) for y in arr]  # list comprehension
                else:
                    ret = [*map(func_reworked, arr)]  # map
                return np.stack(ret, axis=0)

    return inner
