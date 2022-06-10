# src/errors.py

import numpy as np
import traceback
import sys


class InputError(Exception):
    pass


def is_function_vectorised(x, fn, fn_name, fn_input):
    is_fixed = is_vectorised = False
    err_type = 0
    str = 'non_vectorised function'
    _msg_input_error = ("Unable to identify operatorument '{0}'. {1}")

    if isinstance(fn, (int, float)): # fn is a fixed value
        is_fixed = True
        str = 'fixed value'
    elif callable(fn): # fn is a function
        try: # test if fn is vectorised
            fn_ret = fn(x)
            if np.asarray(fn_ret).shape == x.shape:
                is_vectorised = True
            else:
                err_type = 1
        except:
            raise InputError(_msg_input_error.format(fn_name, 'Fail to apply function to data array.'))

        if err_type: # test if fn is executable in non-vectorised mode
            try:
                fn_ret = fn(x[0])
                if not isinstance(fn_ret, (int, float)):
                    raise InputError
            except InputError:
                raise InputError(_msg_input_error.format(fn_name, 'Output must be either a scalar or an array with the same shape as the input.')) from None
            except:
                raise InputError(_msg_input_error.format(fn_name, 'Fail to apply function to data array.')) from None
    else: # fn is neither a scalar nor a function
        err_str = ("Must be one of the following: "
                   "(1) a fixed value (int or float), "
                   "(2) a non-vectorised function that takes {0} as input and a float as output, "
                   "(3) a vectorised function that returns an array of the same shape as the input.")
        raise InputError(_msg_input_error.format(fn_name, err_str.format(fn_input)))
    if is_vectorised:
        str = 'vectorised function'
    return (is_fixed, is_vectorised, str)
