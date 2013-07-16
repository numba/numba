import ctypes
import numpy as np
from llvm import ee as le, passes as lp
from llvm.workaround import avx_support
from . import typing

class JIT(object):
    def __init__(self, lfunc, retty, argtys):
        self.engine = make_engine(lfunc)
        self.lfunc = lfunc
        self.args = argtys
        self.return_type = retty
        self.c_args = [t.ctype_argument() for t in self.args]
        self.c_return_type = (self.return_type.ctype_return()
                              if self.return_type is not None
                              else None)
        self.pointer = self.engine.get_pointer_to_function(lfunc)
        self.callable = make_callable(self.pointer, self.c_return_type,
                                      self.c_args)

    def __call__(self, *args):
        args = [t.ctype_pack_argument(v)
                for t, v in zip(self.args, args)]
        if self.c_return_type is not None:
            ret = self.c_return_type()
            self.callable(*(args + [ctypes.byref(ret)]))
            return self.return_type.ctype_unpack_return(ret)
        else:
            self.callable(*args)

def make_engine(lfunc):
    lmod = lfunc.module

    attrs = []
    if not avx_support.detect_avx_support():
        attrs.append('-avx')

    # NOTE: LLVMPY in Anaconda does not have MCJIT?
    #eb = le.EngineBuilder.new(lmod).mcjit(True).opt(2).mattrs(','.join(attrs))
    eb = le.EngineBuilder.new(lmod).opt(2).mattrs(','.join(attrs))
    tm = eb.select_target()

    # optimize
    pms = lp.build_pass_managers(opt=2, tm=tm, fpm=False)
    pms.pm.run(lmod)
    #print lmod
    
    return eb.create()


def make_callable(ptr, cret, cargs):
    args = list(cargs)
    if cret is not None:
        args += [ctypes.POINTER(cret)]
    prototype = ctypes.CFUNCTYPE(None, *args)
    return prototype(ptr)

