import sys
import math
from numbapro.npm import cgutils, typesets, types

def generate_unary_math(f64name, f32name, callobj, ty):
    nvname = f32name if ty is types.float32 else f64name
    class UnaryMath(object):
        function = callobj, (ty,), ty

        def generic_implement(self, context, args, argtys, retty):
            builder = context.builder
            lty = ty.llvm_as_value()
            lfunc = cgutils.get_function(builder, nvname, lty, [lty])
            return builder.call(lfunc, args)

    return UnaryMath

def generate_binary_math(f64name, f32name, callobj, ty):
    nvname = f32name if ty is types.float32 else f64name
    class BinaryMath(object):
        function = callobj, (ty, ty), ty

        def generic_implement(self, context, args, argtys, retty):
            builder = context.builder
            lty = ty.llvm_as_value()
            lfunc = cgutils.get_function(builder, nvname, lty, [lty, lty])
            return builder.call(lfunc, args)

    return BinaryMath

def generate_powi(nvname, callobj, ty):
    class BinaryMath(object):
        function = callobj, (ty, types.int32), ty

        def generic_implement(self, context, args, argtys, retty):
            builder = context.builder
            lty = ty.llvm_as_value()
            lfunc = cgutils.get_function(builder, nvname, lty, [lty, types.int32])
            return builder.call(lfunc, args)

    return BinaryMath

extensions = []
extensions += [generate_unary_math('__nv_isnand', '__nv_isnanf', math.isnan, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_isinfd', '__nv_isinff', math.isinf, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_ceil', '__nv_ceilf', math.ceil, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_floor', '__nv_floorf', math.floor, ty)
               for ty in typesets.float_set]
extensions += [generate_binary_math('__nv_copysign', '__nv_copysignf', math.copysign, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_fabs', '__nv_fabsf', math.fabs, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_exp', '__nv_expf', math.exp, ty)
               for ty in typesets.float_set]
if sys.version_info[:2] >= (2, 7):
    extensions += [generate_unary_math('__nv_expm1', '__nv_expm1f', math.expm1, ty)
                   for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_sqrt', '__nv_sqrtf', math.sqrt, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_log', '__nv_logf', math.log, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_log10', '__nv_log10f', math.log10, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_log1p', '__nv_log1pf', math.log1p, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_acosh', '__nv_acoshf', math.acosh, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_acos', '__nv_acosf', math.acos, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_cos', '__nv_cosf', math.cos, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_cosh', '__nv_coshf', math.cosh, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_asinh', '__nv_asinhf', math.asinh, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_asin', '__nv_asinf', math.asin, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_sin', '__nv_sinf', math.sin, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_sinh', '__nv_sinhf', math.sinh, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_atan', '__nv_atanf', math.atan, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_atanh', '__nv_atanhf', math.atanh, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_tan', '__nv_tanf', math.tan, ty)
               for ty in typesets.float_set]
extensions += [generate_unary_math('__nv_tanh', '__nv_tanhf', math.tanh, ty)
               for ty in typesets.float_set]
extensions += [generate_binary_math('__nv_atan2', '__nv_atan2f', math.atan2, ty)
               for ty in typesets.float_set]
extensions += [generate_binary_math('__nv_pow', '__nv_powf', math.pow, ty)
               for ty in typesets.float_set]
extensions += [generate_powi('__nv_powi', math.pow, ty)
               for ty in typesets.float_set]
extensions += [generate_binary_math('__nv_fmod', '__nv_fmodf', math.fmod, ty)
               for ty in typesets.float_set]


