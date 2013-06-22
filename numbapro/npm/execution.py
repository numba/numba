import ctypes
import numpy as np
from llvm import ee as le, passes as lp
from llvm.workaround import avx_support
from . import typing

class JIT(object):
    def __init__(self, lfunc, retty, argtys, gvars, globals):
        self.engine = make_engine(lfunc)
        self.args = [to_ctype(x) for x in argtys]
        self.return_type = to_ctype(retty)
        self.pointer = self.engine.get_pointer_to_function(lfunc)
        self.callable = make_callable(self.pointer, self.return_type, self.args)
        self.gvars = gvars

        self.bind_globals(globals)

    def bind_globals(self, globals):
        for name, (ty, gvar) in self.gvars.iteritems():
            cty = to_ctype(ty)
            val = cty(globals[name])
            addr = self.engine.get_pointer_to_global(gvar)
            ptr = ctypes.c_void_p(addr)
            ctypes.cast(ptr, ctypes.POINTER(cty))[0] = val

    def __call__(self, *args):
        args = [prepare_args(t, v)
                for t, v in zip(self.args, args)]
        if self.return_type:
            ret = self.return_type()
            self.callable(*(args + [ctypes.byref(ret)]))
            return prepare_ret(self.return_type, ret)
        else:
            self.callable(*args)


def prepare_args(ty, val):
    if issubclass(ty, (Complex64, Complex128)):
        val = ty(val)
        return ctypes.byref(val)
    elif issubclass(ty, ArrayBase):
        arrayval = ty(data=val.ctypes.data,
                      shape=val.ctypes.shape,
                      strides=val.ctypes.strides)
        return ctypes.byref(arrayval)
    else:
        return ty(val)

def prepare_ret(ty, val):
    if isinstance(val, (Complex64, Complex128)):
        return complex(val.real, val.imag)
    else:
        return val.value

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

def mark_byref(ty):
    if issubclass(ty, (Complex64, Complex128, ArrayBase)):
        return ctypes.POINTER(ty)
    else:
        return ty

def make_callable(ptr, cretty, cargtys):
    args = [mark_byref(ty) for ty in cargtys]
    if cretty:
        args += [ctypes.POINTER(cretty)]
    prototype = ctypes.CFUNCTYPE(None, *args)
    return prototype(ptr)

def to_ctype(ty):
    if ty is None:
        return None
    elif isinstance(ty, typing.ScalarType):
        return to_ctype_scalar(ty)
    elif isinstance(ty, typing.ArrayType):
        return to_ctype_array(ty)
    else:
        raise TypeError(type(ty))

def to_ctype_array(ty):
    return make_array_type(ty.ndim)

def to_ctype_scalar(ty):
    if ty.is_int:
        if ty.is_signed:
            return CTYPE_SIGNED_MAP[ty.bitwidth]
        elif ty.is_unsigned:
            return CTYPE_UNSIGNED_MAP[ty.bitwidth]
    elif ty.is_float:
        return CTYPE_FLOAT_MAP[ty.bitwidth]
    elif ty.is_complex:
        return CTYPE_COMPLEX_MAP[ty.bitwidth]

class Complex64(ctypes.Structure):
    _fields_ = [('real', ctypes.c_float),
                ('imag', ctypes.c_float),]

    def __init__(self, real=0, imag=0):
        if isinstance(real, complex):
            real, imag = real.real, real.imag
        self.real = real
        self.imag = imag

class Complex128(ctypes.Structure):
    _fields_ = [('real', ctypes.c_double),
                ('imag', ctypes.c_double),]

    def __init__(self, real=0, imag=0):
        if isinstance(real, complex):
            real, imag = real.real, real.imag
        self.real = real
        self.imag = imag

class ArrayBase: pass

def make_array_type(nd):
    cache = make_array_type._cache_
    if nd in cache:
        return cache[nd]

    class Array(ArrayBase, ctypes.Structure):
        _fields_ = [('data', ctypes.c_void_p),
                    ('shape', np.ctypeslib.c_intp * nd),
                    ('strides', np.ctypeslib.c_intp * nd),]
    cache[nd] = Array
    return Array

make_array_type._cache_ = {}

CTYPE_SIGNED_MAP = {
     8: ctypes.c_int8,
    16: ctypes.c_int16,
    32: ctypes.c_int32,
    64: ctypes.c_int64,
}

CTYPE_UNSIGNED_MAP = {
     8: ctypes.c_uint8,
    16: ctypes.c_uint16,
    32: ctypes.c_uint32,
    64: ctypes.c_uint64,
}

CTYPE_FLOAT_MAP = {
    32: ctypes.c_float,
    64: ctypes.c_double,
}

CTYPE_COMPLEX_MAP = {
     64: Complex64,
    128: Complex128,
}