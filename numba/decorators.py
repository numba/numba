"""
Contains function decorators and target_registry
"""
from __future__ import print_function, division, absolute_import
import warnings
from . import sigutils
from .targets import registry

# -----------------------------------------------------------------------------
# Decorators


def autojit(*args, **kws):
    """Deprecated.

    Use jit instead.  Calls to jit internally.
    """
    warnings.warn("autojit is deprecated, use jit instead which now performs "
                  "the same functionality", DeprecationWarning)
    return jit(*args, **kws)


def jit(signature_or_function=None, argtypes=None, restype=None, locals={},
        target='cpu', **targetoptions):
    """jit([signature_or_function, [locals={}, [target='cpu', [**targetoptions]]]])

    This function is used to compile a Python function into native code. It is
    designed to be used as a decorator for the function to be compiled,
    but it can also be called as a regular function.
    
    Args
    -----
    signature_or_function: function or str
        This argument takes either the function to be compiled, or the signature
        of the function to be compiled. If this function is used as a decorator,
        the function to be compiled is the decorated function. In that case,
        this argument should only be used to optionally specify the function
        signature. If this function is called like a regular function, and this
        argument is used to specify the function signature, this function will
        return another jit function object which can be called again with the
        function to be compiled as this argument.

    argtypes: deprecated

    restype: deprecated

    locals: dict
        Mapping of local variable names to Numba types. Used to override the
        types deduced by Numba's type inference engine.

    targets: str
        Specifies the target platform to compile for. Valid targets are cpu,
        gpu, npyufunc, and cuda. Defaults to cpu.

    targetoptions: 
        For a cpu target, valid options are:
            nopython: bool
                Set to True to disable the use of PyObjects and Python API
                calls. The default behavior is to allow the use of PyObjects
                and Python API. Default value is False.

            forceobj: bool
                Set to True to force the use of PyObjects for every value.
                Default value is False.

            looplift: bool
                Set to True to enable jitting loops in nopython mode while
                leaving surrounding code in object mode. This allows functions
                to allocate NumPy arrays and use Python objects, while the
                tight loops in the function can still be compiled in nopython
                mode. Any arrays that the tight loop uses should be created
                before the loop is entered. Default value is True.

            wraparound: bool
                Set to True to enable array indexing wraparound for negative
                indices, for a small performance penalty. Default value
                is True.

    Returns
    --------

    compiled function

    Examples
    --------
    The function can be used in the following ways:

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

    """

    # Handle deprecated argtypes and restype keyword arguments
    if argtypes is not None:

        assert signature_or_function is None, "argtypes used but " \
                                              "signature is provided"
        warnings.warn("Keyword argument 'argtypes' is deprecated",
                      DeprecationWarning)
        if restype is None:
            signature_or_function = tuple(argtypes)
        else:
            signature_or_function = restype(*argtypes)

    # Handle signature
    if signature_or_function is None:
        # Used as autojit
        def configured_jit(arg):
            return jit(arg, locals=locals, target=target, **targetoptions)
        return configured_jit
    elif sigutils.is_signature(signature_or_function):
        # Function signature is provided
        sig = signature_or_function
        return _jit(sig, locals=locals, target=target,
                    targetoptions=targetoptions)
    else:
        # No signature is provided
        pyfunc = signature_or_function
        dispatcher = registry.target_registry[target]
        dispatcher = dispatcher(py_func=pyfunc, locals=locals,
                                targetoptions=targetoptions)
        # NOTE This affects import time for large function
        # # Compile a pure object mode
        # if target == 'cpu' and not targetoptions.get('nopython', False):
        #     dispatcher.compile((), locals=locals, forceobj=True)
        return dispatcher


def _jit(sig, locals, target, targetoptions):
    dispatcher = registry.target_registry[target]

    def wrapper(func):
        disp = dispatcher(py_func=func,  locals=locals,
                          targetoptions=targetoptions)
        disp.compile(sig)
        disp.disable_compile()
        return disp

    return wrapper


def njit(*args, **kws):
    """
    Equivalent to jit(nopython=True)

    See documentation for jit function/decorator for full description.
    """
    if 'nopython' in kws:
        warnings.warn('nopython is set for njit and is ignored', RuntimeWarning)
    if 'forceobj' in kws:
        warnings.warn('forceobj is set for njit and is ignored', RuntimeWarning)
    kws.update({'nopython': True})
    return jit(*args, **kws)


