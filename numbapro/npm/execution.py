import ctypes
from llvm import ee as le, passes as lp
from llvm.workaround import avx_support
from . import types

class JIT(object):
    def __init__(self, lfunc, retty, argtys, exceptions):
        self.engine = make_engine(lfunc)
        self.lfunc = lfunc
        self.args = argtys
        self.return_type = retty
        self.c_args = [t.ctype_as_argument() for t in self.args]
        self.c_return_type = (self.return_type.ctype_as_return()
                              if self.return_type != types.void
                              else None)
        self.pointer = self.engine.get_pointer_to_function(lfunc)
        self.callable = make_callable(self.pointer, self.c_return_type,
                                      self.c_args)
        self.exceptions = exceptions
    
    def __call__(self, *args):
        args = [t.ctype_pack_argument(v)
                for t, v in zip(self.args, args)]
        if self.c_return_type is not None:
            # has return value
            ret = self.c_return_type()
            errcode = self.callable(*(args + [ctypes.byref(ret)]))
            if errcode == 0:
                return self.return_type.ctype_unpack_return(ret)
        else:
            # no return value
            errcode = self.callable(*args)
            if errcode == 0:
                return
        # exception handling
        try:
            einfo = self.exceptions[errcode]
            raise einfo.exc
        except KeyError:
            raise RuntimeError('an unknown exception has raised: errcode=%d' %
                                errcode)

def make_engine(lfunc):
    lmod = lfunc.module

    attrs = []
    if not avx_support.detect_avx_support():
        attrs.append('-avx')

    # NOTE: LLVMPY in Anaconda does not have MCJIT?
    #eb = le.EngineBuilder.new(lmod).mcjit(True).opt(2).mattrs(','.join(attrs))
    eb = le.EngineBuilder.new(lmod).opt(3).mattrs(','.join(attrs))
    tm = eb.select_target()

    # optimize
    pms = lp.build_pass_managers(opt=3, tm=tm, fpm=False,
                                 loop_vectorize=True)
    pms.pm.run(lmod)

    return eb.create()

def make_callable(ptr, cret, cargs):
    args = list(cargs)
    if cret is not None:
        args += [ctypes.POINTER(cret)]
    prototype = ctypes.CFUNCTYPE(ctypes.c_int, *args)
    return prototype(ptr)

