"""
Stub for early user testing
"""
from __future__ import print_function, division, absolute_import
import warnings
from numba import types, dispatcher, utils

target_registry = utils.UniqueDict()
target_registry['cpu'] = dispatcher.Overloaded
target_registry['nopython-cpu'] = dispatcher.NPMOverloaded


def autojit(*args, **kws):
    warnings.warn("autojit is deprecated, use jit instead which now performs "
                  "the same functionality", DeprecationWarning)
    return jit(*args, **kws)


def jit(*args, **kws):
    """jit(signature_or_function, [nopython=False, [target='cpu']])
    The function can be used as the following versions:

    1) jit(function) -> Overloaded
    2) jit(signature, [nopython=False, [target='cpu']]) -> jit(function)

    It is used as a decorator to create low-level functions that is
    overloaded for different signatures.

    If a signature is not given (version 1):

    @jit
    def foo(x, y):
        ...

    It converts ``foo`` into an ``Overloaded`` object that can generate
    specialized machine code for the given arguments when called as a normal
    function.  The generated code is reused the next time.

    If a signature is given (version 2):

    @jit(int32(int32, int32))
    def foo(x, y):
        ...

    which is equivalent to calling ``foo.jit()`` with the same arguments.  This
    forces the given signature to be compiled.

    The keyword arguments are compile flags.

    - nopython: bool
        Set to True to forbids the use of Python API.
    - target: str
        Control code generation target.  Different target may return different
        objects.  The above information is guarantee to be True for "cpu"
        target.
    """
    if not args:
        def configured_jit(arg):
            return jit(arg, **kws)
        return configured_jit
    elif dispatcher.is_signature(args[0]):
        # Function signature is provided
        [sig] = args
        return _jit(sig, kws)
    else:
        # No signature is provided
        [pyfunc] = args
        dispcls = target_registry[kws.pop('target', 'cpu')]
        return dispcls(py_func=pyfunc)


def _jit(sig, kws):
    dispcls = target_registry[kws.pop('target', 'cpu')]

    def wrapper(func):
        disp = dispcls(py_func=func)
        disp.compile(sig, **kws)
        return disp

    return wrapper

