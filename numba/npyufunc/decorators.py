from __future__ import print_function, division, absolute_import
from .ufuncbuilder import UFuncBuilder, GUFuncBuilder

from numba import utils


class Vectorize(object):
    target_registry = utils.UniqueDict({'nopython-cpu': UFuncBuilder})

    def __new__(cls, func, **kws):
        target = kws.pop('target', 'nopython-cpu')
        try:
            imp = cls.target_registry[target]
        except KeyError:
            raise ValueError("Unsupported target: %s" % target)

        return imp(func, kws)


class GUVectorize(object):
    target_registry = utils.UniqueDict({'nopython-cpu': GUFuncBuilder})

    def __new__(cls, func, signature, **kws):
        target = kws.pop('target', 'nopython-cpu')
        try:
            imp = cls.target_registry[target]
        except KeyError:
            raise ValueError("Unsupported target: %s" % target)

        return imp(func, signature, kws)


def vectorize(ftylist, **kws):
    """vectorize(ftylist[, target='nopython-cpu', [**kws]])

    A decorator to create numpy ufunc object from Numba compiled code.

    Args
    -----
    ftylist: iterable
        An iterable of type signatures, which are either
        function type object or a string describing the
        function type.

    target: str
            A string for code generation target.  Default to "nopython-cpu".

    Returns
    --------

    A NumPy universal function

    Example
    -------
        @vectorize(['float32(float32, float32)',
                    'float64(float64, float64)'])
        def sum(a, b):
            return a + b

    """
    if isinstance(ftylist, str):
        # Common user mistake
        ftylist = [ftylist]

    def wrap(func):
        vec = Vectorize(func, **kws)
        for fty in ftylist:
            vec.add(fty)
        return vec.build_ufunc()
    return wrap


def guvectorize(ftylist, signature, **kws):
    """guvectorize(ftylist, signature, [, target='cpu', [**kws]])

    A decorator to create numpy generialized-ufunc object from Numba compiled
    code.

    Args
    -----
    ftylist: iterable
        An iterable of type signatures, which are either
        function type object or a string describing the
        function type.


    signature: str
        A NumPy generialized-ufunc signature.
        e.g. "(m, n), (n, p)->(m, p)"

    target: str
            A string for code generation target.  Default to "cpu"
            this should be

    Returns
    --------

    A NumPy generialized universal-function

    Example
    -------
        @guvectorize(['void(int32[:,:], int32[:,:], int32[:,:])',
                      'void(float32[:,:], float32[:,:], float32[:,:])'],
                      '(x, y),(x, y)->(x, y)')
        def add_2d_array(a, b):
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    c[i, j] = a[i, j] + b[i, j]

    """
    if isinstance(ftylist, str):
        # Common user mistake
        ftylist = [ftylist]

    def wrap(func):
        guvec = GUVectorize(func, signature, **kws)
        for fty in ftylist:
            guvec.add(fty)
        return guvec.build_ufunc()
    return wrap


