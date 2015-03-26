from __future__ import absolute_import, print_function, division
import numpy

from numba import jit, typeof, utils
from . import _internal, ufuncbuilder

class DUFunc(_internal._DUFunc):
    def __init__(self, py_func, **kws):
        dispatcher = jit(target='npyufunc')(py_func)
        super(DUFunc, self).__init__(dispatcher, **kws)
        # Clean up keyword arguments, and assume everything else are
        # compiler target options.
        kws.pop('identity', None)
        kws.pop('nin', None)
        kws.pop('nout', None)
        self.targetoptions = kws

    def _compile_and_invoke(self, *args, **kws):
        assert len(kws) == 0
        ewise_sig = tuple(typeof(arg).dtype for arg in args)
        #import pdb; pdb.set_trace()
        cres, args, return_type = ufuncbuilder._compile_ewise_function(
            self.dispatcher, self.targetoptions, ewise_sig)
        sig = ufuncbuilder._check_ufunc_signature(cres, args, return_type)
        dtypenums, ptr, env = ufuncbuilder._build_ewise_ufunc_wrapper(cres, sig)
        self._add_loop(numpy.dtype(numpy.void).num, utils.longint(ptr),
                       dtypenums)
        return self.ufunc(*args, **kws)
