from __future__ import absolute_import, print_function

from numba import utils


def generate_implementation(py_func, impl_kind, *args, **kws):
    """
    Generate a function implementation based on whether a
    ``jit`` or ``generated_jit`` decorator has been applied.

    When ``impl_kind`` is 'direct', the function is simply returned.
    By contrast, a 'generated'  function returns an implementation
    based on the numba types supplied in *args and **kws.

    Parameters
    ----------
    py_func : callable
        Function
    impl_kind : {'direct', 'generated'}
        If 'direct',  ``py_func`` is simply returned.

    Returns
    -------
    callable
        The (possibly) generated function

    """
    if impl_kind == "direct":
        return py_func
    elif impl_kind == "generated":
        impl = py_func(*args, **kws)
        # Check the generating function and implementation signatures are
        # compatible, otherwise compiling would fail later.
        pysig = utils.pysignature(py_func)
        implsig = utils.pysignature(impl)
        ok = len(pysig.parameters) == len(implsig.parameters)
        if ok:
            for pyparam, implparam in zip(pysig.parameters.values(),
                                          implsig.parameters.values()):
                # We allow the implementation to omit default values, but
                # if it mentions them, they should have the same value...
                if (pyparam.name != implparam.name or
                    pyparam.kind != implparam.kind or
                    (implparam.default is not implparam.empty and
                     implparam.default != pyparam.default)):
                    ok = False
        if not ok:
            raise TypeError("generated implementation %s should be compatible "
                            "with signature '%s', but has signature '%s'"
                            % (impl, pysig, implsig))
        return impl

    raise ValueError("Invalid implementation kind %s" % impl_kind)
