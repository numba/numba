from __future__ import print_function, division, absolute_import
import inspect

from . import _internal, dufunc
from .ufuncbuilder import GUFuncBuilder
from .parallel import ParallelUFuncBuilder, ParallelGUFuncBuilder

from numba.targets.registry import TargetRegistry


class _BaseVectorize(object):

    @classmethod
    def get_identity(cls, kwargs):
        return kwargs.pop('identity', None)

    @classmethod
    def get_cache(cls, kwargs):
        return kwargs.pop('cache', False)

    @classmethod
    def get_target_implementation(cls, kwargs):
        target = kwargs.pop('target', 'cpu')
        try:
            return cls.target_registry[target]
        except KeyError:
            raise ValueError("Unsupported target: %s" % target)


class Vectorize(_BaseVectorize):
    target_registry = TargetRegistry({'cpu': dufunc.DUFunc,
                                      'parallel': ParallelUFuncBuilder,})

    def __new__(cls, func, **kws):
        identity = cls.get_identity(kws)
        cache = cls.get_cache(kws)
        imp = cls.get_target_implementation(kws)
        return imp(func, identity=identity, cache=cache, targetoptions=kws)


class GUVectorize(_BaseVectorize):
    target_registry = TargetRegistry({'cpu': GUFuncBuilder,
                                      'parallel': ParallelGUFuncBuilder,})

    def __new__(cls, func, signature, **kws):
        identity = cls.get_identity(kws)
        cache = cls.get_cache(kws)
        imp = cls.get_target_implementation(kws)
        return imp(func, signature, identity=identity, cache=cache,
                   targetoptions=kws)


def vectorize(ftylist_or_function=(), **kws):
    """vectorize(ftylist_or_function=(), target='cpu', identity=None, **kws)

    A decorator that creates a Numpy ufunc object using Numba compiled
    code.  When no arguments or only keyword arguments are given,
    vectorize will return a Numba dynamic ufunc (DUFunc) object, where
    compilation/specialization may occur at call-time.

    Args
    -----
    ftylist_or_function: function or iterable

        When the first argument is a function, signatures are dealt
        with at call-time.

        When the first argument is an iterable of type signatures,
        which are either function type object or a string describing
        the function type, signatures are finalized at decoration
        time.

    Keyword Args
    ------------

    target: str
            A string for code generation target.  Default to "cpu".

    identity: int, str, or None
        The identity (or unit) value for the element-wise function
        being implemented.  Allowed values are None (the default), 0, 1,
        and "reorderable".

    cache: bool
        Turns on caching.


    Returns
    --------

    A NumPy universal function

    Examples
    -------
        @vectorize(['float32(float32, float32)',
                    'float64(float64, float64)'], identity=1)
        def sum(a, b):
            return a + b

        @vectorize
        def sum(a, b):
            return a + b

        @vectorize(identity=1)
        def mul(a, b):
            return a * b

    """
    if isinstance(ftylist_or_function, str):
        # Common user mistake
        ftylist = [ftylist_or_function]
    elif inspect.isfunction(ftylist_or_function):
        return dufunc.DUFunc(ftylist_or_function, **kws)
    elif ftylist_or_function is not None:
        ftylist = ftylist_or_function

    def wrap(func):
        vec = Vectorize(func, **kws)
        for sig in ftylist:
            vec.add(sig)
        if len(ftylist) > 0:
            vec.disable_compile()
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

    cache: bool
        Turns on caching.

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
        def add_2d_array(a, b, c):
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
