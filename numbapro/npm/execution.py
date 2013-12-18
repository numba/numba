import ctypes
import llvm.ee as le
import llvm.passes as lp
from llvm.workaround import avx_support
from . import types


class JIT(object):
    __slots__ = ('engine', 'lfunc', 'args', 'return_type', 'c_args',
                 'c_return_type', 'pointer', 'callable', 'exceptions')

    def __init__(self, lfunc, retty, argtys, exceptions):
        self.engine = EngineManager().get_engine(lfunc)
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

    def __del__(self):
        self.engine.remove_module(self.lfunc.module)

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


class EngineManager(object):
    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            inst = object.__new__(cls)
            cls._singleton = inst
            return inst
        else:
            return cls._singleton

    def __init__(self):
        self.engine = None
        self.pm = None
        self.tm = None

    def make_engine(self, module):
        attrs = []
        if not avx_support.detect_avx_support():
            attrs.append('-avx')

        eb = le.EngineBuilder.new(module).opt(3).mattrs(','.join(attrs))
        self.tm = eb.select_target()

        self.pm = self.make_pm()
        self.pm.run(module)

        engine = eb.create(self.tm)
        return engine

    def make_pm(self):
        pms = lp.build_pass_managers(opt=2, tm=self.tm, fpm=False,
                                     loop_vectorize=True)
        return pms.pm

    def get_engine(self, lfunc):
        mod = lfunc.module
        if self.engine is None:
            self.engine = self.make_engine(mod)
        else:
            self.pm.run(mod)                # run optimizer
            self.engine.add_module(mod)   # JIT module
        return self.engine


def make_callable(ptr, cret, cargs):
    args = list(cargs)
    if cret is not None:
        args += [ctypes.POINTER(cret)]
    prototype = ctypes.CFUNCTYPE(ctypes.c_int, *args)
    return prototype(ptr)

