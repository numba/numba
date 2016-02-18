from numba.extending import (typeof_impl, type_callable,
                             lower_builtin, lower_cast,
                             models, register_model,
                             box, unbox, reflect, NativeValue,
                             make_attribute_wrapper,
                             )
from numba import types, cgutils
from .imputils import impl_ret_borrowed

@type_callable('__array_wrap__')
def type_array_wrap(context):
    def typer(input_type, result):
        if isinstance(input_type, types.SmartArrayType):
            return input_type.copy(dtype=result.dtype,
                                   ndim=result.ndim,
                                   layout=result.layout)

    return typer

@lower_builtin('__array__', types.SmartArrayType)
def array_as_array(context, builder, sig, args):
    [argtype], [arg] = sig.args, args
    val = cgutils.create_struct_proxy(argtype)(context, builder, ref=arg)
    return val._get_ptr_by_name('data')

@lower_builtin('__array_wrap__', types.SmartArrayType, types.Array)
def array_wrap_array(context, builder, sig, args):
    dest = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    dest.data = args[1]
    return impl_ret_borrowed(context, builder, sig.return_type, dest._getvalue())

