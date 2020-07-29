import sys
import math
from llvmlite.llvmpy.core import Type
from numba.core import types, cgutils
from numba.core.imputils import Registry

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


booleans = []
booleans += [('__nv_isnand', '__nv_isnanf', math.isnan)]
booleans += [('__nv_isinfd', '__nv_isinff', math.isinf)]

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
