import math
import operator
from llvmlite.llvmpy.core import Type
from numba.core import types, cgutils
from numba.core.imputils import Registry
from numba.core.types import float32, complex64

registry = Registry()
lower = registry.lower

float_set = types.float32, types.float64


def bool_implement(nvname, ty):
    def core(context, builder, sig, args):
        assert sig.return_type == types.boolean, nvname
        fty = context.get_value_type(ty)
        lmod = builder.module
        fnty = Type.function(Type.int(), [fty])
        fn = lmod.get_or_insert_function(fnty, name=nvname)
        result = builder.call(fn, args)
        return context.cast(builder, result, types.int32, types.boolean)

    return core


def unary_implement(nvname, ty):
    def core(context, builder, sig, args):
        fty = context.get_value_type(ty)
        lmod = builder.module
        fnty = Type.function(fty, [fty])
        fn = lmod.get_or_insert_function(fnty, name=nvname)
        return builder.call(fn, args)

    return core


def binary_implement(nvname, ty):
    def core(context, builder, sig, args):
        fty = context.get_value_type(ty)
        lmod = builder.module
        fnty = Type.function(fty, [fty, fty])
        fn = lmod.get_or_insert_function(fnty, name=nvname)
        return builder.call(fn, args)

    return core


def powi_implement(nvname):
    def core(context, builder, sig, args):
        [base, pow] = args
        [basety, powty] = sig.args
        lmod = builder.module
        fty = context.get_value_type(basety)
        ity = context.get_value_type(types.int32)
        fnty = Type.function(fty, [fty, ity])
        fn = lmod.get_or_insert_function(fnty, name=nvname)
        return builder.call(fn, [base, pow])

    return core


lower(math.pow, types.float32, types.int32)(powi_implement('__nv_powif'))
lower(math.pow, types.float64, types.int32)(powi_implement('__nv_powi'))


def frexp_implement(nvname):
    def core(context, builder, sig, args):
        fracty, expty = sig.return_type
        float_type = context.get_value_type(fracty)
        int_type = context.get_value_type(expty)
        fnty = Type.function(float_type, [float_type, Type.pointer(int_type)])

        fn = builder.module.get_or_insert_function(fnty, name=nvname)
        expptr = cgutils.alloca_once(builder, int_type, name='exp')

        ret = builder.call(fn, (args[0], expptr))
        return cgutils.pack_struct(builder, (ret, builder.load(expptr)))

    return core


lower(math.frexp, types.float32)(frexp_implement('__nv_frexpf'))
lower(math.frexp, types.float64)(frexp_implement('__nv_frexp'))

lower(math.ldexp, types.float32, types.int32)(powi_implement('__nv_ldexpf'))
lower(math.ldexp, types.float64, types.int32)(powi_implement('__nv_ldexp'))


booleans = []
booleans += [('__nv_isnand', '__nv_isnanf', math.isnan)]
booleans += [('__nv_isinfd', '__nv_isinff', math.isinf)]
booleans += [('__nv_isfinited', '__nv_finitef', math.isfinite)]

unarys = []
unarys += [('__nv_ceil', '__nv_ceilf', math.ceil)]
unarys += [('__nv_floor', '__nv_floorf', math.floor)]
unarys += [('__nv_fabs', '__nv_fabsf', math.fabs)]
unarys += [('__nv_exp', '__nv_expf', math.exp)]
unarys += [('__nv_expm1', '__nv_expm1f', math.expm1)]
unarys += [('__nv_erf', '__nv_erff', math.erf)]
unarys += [('__nv_erfc', '__nv_erfcf', math.erfc)]
unarys += [('__nv_tgamma', '__nv_tgammaf', math.gamma)]
unarys += [('__nv_lgamma', '__nv_lgammaf', math.lgamma)]
unarys += [('__nv_sqrt', '__nv_sqrtf', math.sqrt)]
unarys += [('__nv_log', '__nv_logf', math.log)]
unarys += [('__nv_log10', '__nv_log10f', math.log10)]
unarys += [('__nv_log1p', '__nv_log1pf', math.log1p)]
unarys += [('__nv_acosh', '__nv_acoshf', math.acosh)]
unarys += [('__nv_acos', '__nv_acosf', math.acos)]
unarys += [('__nv_cos', '__nv_cosf', math.cos)]
unarys += [('__nv_cosh', '__nv_coshf', math.cosh)]
unarys += [('__nv_asinh', '__nv_asinhf', math.asinh)]
unarys += [('__nv_asin', '__nv_asinf', math.asin)]
unarys += [('__nv_sin', '__nv_sinf', math.sin)]
unarys += [('__nv_sinh', '__nv_sinhf', math.sinh)]
unarys += [('__nv_atan', '__nv_atanf', math.atan)]
unarys += [('__nv_atanh', '__nv_atanhf', math.atanh)]
unarys += [('__nv_tan', '__nv_tanf', math.tan)]
unarys += [('__nv_tanh', '__nv_tanhf', math.tanh)]

binarys = []
binarys += [('__nv_copysign', '__nv_copysignf', math.copysign)]
binarys += [('__nv_atan2', '__nv_atan2f', math.atan2)]
binarys += [('__nv_pow', '__nv_powf', math.pow)]
binarys += [('__nv_fmod', '__nv_fmodf', math.fmod)]
binarys += [('__nv_hypot', '__nv_hypotf', math.hypot)]


for name64, name32, key in booleans:
    impl64 = bool_implement(name64, types.float64)
    lower(key, types.float64)(impl64)
    impl32 = bool_implement(name32, types.float32)
    lower(key, types.float32)(impl32)


for name64, name32, key in unarys:
    impl64 = unary_implement(name64, types.float64)
    lower(key, types.float64)(impl64)
    impl32 = unary_implement(name32, types.float32)
    lower(key, types.float32)(impl32)

for name64, name32, key in binarys:
    impl64 = binary_implement(name64, types.float64)
    lower(key, types.float64, types.float64)(impl64)
    impl32 = binary_implement(name32, types.float32)
    lower(key, types.float32, types.float32)(impl32)


def modf_implement(nvname, ty):
    def core(context, builder, sig, args):
        arg, = args
        argty, = sig.args
        fty = context.get_value_type(argty)
        lmod = builder.module
        ptr = cgutils.alloca_once(builder, fty)
        fnty = Type.function(fty, [fty, fty.as_pointer()])
        fn = lmod.get_or_insert_function(fnty, name=nvname)
        out = builder.call(fn, [arg, ptr])
        ret = context.make_tuple(builder, types.UniTuple(argty, 2),
                                 [out, builder.load(ptr)])
        return ret
    return core


for (ty, intrin) in ((types.float64, '__nv_modf',),
                     (types.float32, '__nv_modff',)):
    lower(math.modf, ty)(modf_implement(intrin, ty))


# Complex power implementations - translations of _Py_c_pow from CPython
# https://github.com/python/cpython/blob/a755410e054e1e2390de5830befc08fe80706c66/Objects/complexobject.c#L123-L151
#
# The complex64 variant casts all constants and some variables to ensure that
# as much computation is done in single precision as possible. A small number
# of operations are still done in 64-bit, but these come from libdevice code.

@lower(operator.pow, types.complex64, types.complex64)
@lower(operator.ipow, types.complex64, types.complex64)
@lower(pow, types.complex64, types.complex64)
def cpow_impl(context, builder, sig, args):
    def cuda_cpowf(a, b):

        if b.real == float32(0.0) and b.imag == float32(0.0):
            return complex64(1.0) + complex64(0.0j)
        elif a.real == float32(0.0) and b.real == float32(0.0):
            return complex64(0.0) + complex64(0.0j)

        vabs = math.hypot(a.real, a.imag)
        len = math.pow(vabs, b.real)
        at = math.atan2(a.imag, a.real)
        phase = at * b.real
        if b.imag != float32(0.0):
            len /= math.exp(at * b.imag)
            phase += b.imag * math.log(vabs)

        return len * (complex64(math.cos(phase)) +
                      complex64(math.sin(phase) * complex64(1.0j)))

    return context.compile_internal(builder, cuda_cpowf, sig, args)

@lower(operator.pow, types.complex128, types.complex128)
@lower(operator.ipow, types.complex128, types.complex128)
@lower(pow, types.complex128, types.complex128)
def cpow_impl(context, builder, sig, args):
    def cuda_cpow(a, b):

        if b.real == 0.0 and b.imag == 0.0:
            return 1.0 + 0.0j
        elif a.real == 0.0 and b.real == 0.0:
            return 0.0 + 0.0j

        vabs = math.hypot(a.real, a.imag)
        len = math.pow(vabs, b.real)
        at = math.atan2(a.imag, a.real)
        phase = at * b.real
        if b.imag != 0.0:
            len /= math.exp(at * b.imag)
            phase += b.imag * math.log(vabs)

        return len * (math.cos(phase) + math.sin(phase) * 1.0j)

    return context.compile_internal(builder, cuda_cpow, sig, args)
