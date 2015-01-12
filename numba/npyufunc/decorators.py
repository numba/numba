from __future__ import print_function, division, absolute_import

from . import _internal
from .ufuncbuilder import UFuncBuilder, GUFuncBuilder

from numba.targets.registry import TargetRegistry


class _BaseVectorize(object):

    @classmethod
    def get_identity(cls, kwargs):
        return kwargs.pop('identity', None)

    @classmethod
    def get_target_implementation(cls, kwargs):
        target = kwargs.pop('target', 'cpu')
        try:
            return cls.target_registry[target]
        except KeyError:
            raise ValueError("Unsupported target: %s" % target)


class Vectorize(_BaseVectorize):
    target_registry = TargetRegistry({'cpu': UFuncBuilder})

    def __new__(cls, func, **kws):
        identity = cls.get_identity(kws)
        imp = cls.get_target_implementation(kws)
        return imp(func, identity, kws)


class GUVectorize(_BaseVectorize):
    target_registry = TargetRegistry({'cpu': GUFuncBuilder})

    def __new__(cls, func, signature, **kws):
        identity = cls.get_identity(kws)
        imp = cls.get_target_implementation(kws)
        return imp(func, signature, identity, kws)


def vectorize(ftylist, **kws):
    """vectorize(ftylist, target='cpu', identity=None, **kws)

    A decorator to create numpy ufunc object from Numba compiled code.

    Args
    -----
    ftylist: iterable
        An iterable of type signatures, which are either
        function type object or a string describing the
        function type.

    target: str
            A string for code generation target.  Default to "cpu".

    identity: int, str, or None
        The identity (or unit) value for the element-wise function
        being implemented.  Allowed values are None (the default), 0, 1,
        and "reorderable".

    Returns
    --------

    A NumPy universal function

    Example
    -------
        @vectorize(['float32(float32, float32)',
                    'float64(float64, float64)'], identity=1)
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
    """guvectorize(ftylist, signature, target='cpu', identity=None, **kws)

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

    identity: int, str, or None
        The identity (or unit) value for the element-wise function
        being implemented.  Allowed values are None (the default), 0, 1,
        and "reorderable".

    target: str
            A string for code generation target.  Defaults to "cpu".

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


