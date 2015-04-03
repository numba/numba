from __future__ import absolute_import, print_function, division
import numpy

from numba import jit, typeof, utils
from . import _internal, ufuncbuilder

class DUFunc(_internal._DUFunc):
    def __init__(self, py_func, **kws):
        dispatcher = jit(target='npyufunc')(py_func)
        super(DUFunc, self).__init__(dispatcher, **kws)
        # Clean up _DUFunc keyword arguments, and assume anything that
        # remains is a compiler option.
        kws.pop('identity', None)
        kws.pop('nin', None)
        kws.pop('nout', None)
        self.targetoptions = kws

    def _compile_for_args(self, *args, **kws):
        assert len(kws) == 0
        argtys = tuple(typeof(arg) for arg in args)
        if all(hasattr(argty, 'dtype') for argty in argtys):
            argtys = tuple(argty.dtype for argty in argtys)
        cres, args, return_type = ufuncbuilder._compile_ewise_function(
            self.dispatcher, self.targetoptions, argtys)
        sig = ufuncbuilder._check_ufunc_signature(cres, args, return_type)
        dtypenums, ptr, env = ufuncbuilder._build_ewise_ufunc_wrapper(cres, sig)
        self._add_loop(utils.longint(ptr), dtypenums)
        self.keepalive.append((ptr, cres.library, env))
        return
