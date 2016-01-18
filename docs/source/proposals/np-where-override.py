import numpy as np

from numba import types
from numba.extending import overload

@overload(np.where)
def where(cond, x, y):
    """
    Implement np.where().
    """
    # Choose implementation based on argument types.
    if isinstance(cond, types.Array):
        # Array where() => return an array of the same shape
        if all(ty.layout == 'C' for ty in (cond, x, y)):
            def where_impl(cond, x, y):
                """
                Fast implementation for C-contiguous arrays
                """
                shape = cond.shape
                if x.shape != shape or y.shape != shape:
                    raise ValueError("all inputs should have the same shape")
                res = np.empty_like(x)
                cf = cond.flat
                xf = x.flat
                yf = y.flat
                rf = res.flat
                for i in range(cond.size):
                    rf[i] = xf[i] if cf[i] else yf[i]
                return res
        else:
            def where_impl(cond, x, y):
                """
                Generic implementation for other arrays
                """
                shape = cond.shape
                if x.shape != shape or y.shape != shape:
                    raise ValueError("all inputs should have the same shape")
                res = np.empty_like(x)
                for idx, c in np.ndenumerate(cond):
                    res[idx] = x[idx] if c else y[idx]
                return res

    else:
        def where_impl(cond, x, y):
            """
            Scalar where() => return a 0-dim array
            """
            scal = x if cond else y
            return np.full_like(scal, scal)

    return where_impl
