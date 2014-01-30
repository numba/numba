"""
Contains function decorators and target_registry
"""
from __future__ import print_function, division, absolute_import
import warnings
from numba import sigutils
from numba.targets import registry

# -----------------------------------------------------------------------------
# Decorators


def autojit(*args, **kws):
    """Deprecated.

    Use jit instead.  Calls to jit internally.
    """
    warnings.warn("autojit is deprecated, use jit instead which now performs "
                  "the same functionality", DeprecationWarning)
    return jit(*args, **kws)


def jit(signature_or_function=None, target='cpu', **targetoptions):
    """jit([signature_or_function, [target='cpu', [**targetoptions]]])

    The function can be used as the following versions:

    1) jit(signature, [target='cpu', [**targetoptions]]) -> jit(function)

        Equivalent to:

            d = dispatcher(function, targetoptions)
            d.compile(signature)

        Create a dispatcher object for a python function and default
        target-options.  Then, compile the funciton with the given signature.

        Example:

            @jit("void(int32, float32)")
            def foo(x, y):
                return x + y

    2) jit(function) -> dispatcher

        Same as old autojit.  Create a dispatcher function object that
        specialize at call site.

        Example:

            @jit
            def foo(x, y):
                return x + y

    3) jit([target='cpu', [**targetoptions]]) -> configured_jit(function)

        Same as old autojit and 2).  But configure with target and default
        target-options.


        Example:

            @jit(target='cpu', nopython=True)
            def foo(x, y):
                return x + y

    Target Options
    ---------------
    The CPU (default target) defines the following:

        - nopython: [bool]

            Set to True to disable the use of PyObjects and Python API
            calls.  The default behavior is to allow the use of PyObjects and
            Python API.  Default value is False.

        - forceobj: [bool]

            Set to True to force the use of PyObjects for every value.  Default
            value is False.

    """
    if signature_or_function is None:
        # Used as autojit
        def configured_jit(arg):
            return jit(arg, target=target, **targetoptions)
        return configured_jit
    elif sigutils.is_signature(signature_or_function):
        # Function signature is provided
        sig = signature_or_function
        return _jit(sig, target=target, targetoptions=targetoptions)
    else:
        # No signature is provided
        pyfunc = signature_or_function
        dispatcher = registry.target_registry[target]
        return dispatcher(py_func=pyfunc, targetoptions=targetoptions)


def _jit(sig, target, targetoptions):
    dispatcher = registry.target_registry[target]

    def wrapper(func):
        disp = dispatcher(py_func=func, targetoptions=targetoptions)
        disp.compile(sig)
        return disp

    return wrapper

