from numba.core import types
from numba.core.imputils import lower_cast


@lower_cast(types.NumPyIntegerLiteral, types.NumPyInteger)
def literal_int_to_number(context, builder, fromty, toty, val):
    lit = context.get_constant_generic(builder,
                                       fromty.literal_type,
                                       fromty.literal_value,)
    return context.cast(builder, lit, fromty.literal_type, toty)


@lower_cast(types.NumPyInteger, types.NumPyInteger)
def np_int_to_np_int(context, builder, fromty, toty, val):
    if toty.bitwidth == fromty.bitwidth:
        # Just a change of signedness
        return val
    elif toty.bitwidth < fromty.bitwidth:
        # Downcast
        return builder.trunc(val, context.get_value_type(toty))
    elif fromty.signed:
        # Signed upcast
        return builder.sext(val, context.get_value_type(toty))
    else:
        # Unsigned upcast
        return builder.zext(val, context.get_value_type(toty))


@lower_cast(types.PythonInteger, types.NumPyInteger)
def py_int_to_np_int(context, builder, fromty, toty, val):
    if toty.bitwidth == fromty.bitwidth:
        # Just a change of signedness
        return val
    elif toty.bitwidth < fromty.bitwidth:
        # Downcast
        return builder.trunc(val, context.get_value_type(toty))
    elif fromty.signed:
        # Signed upcast
        return builder.sext(val, context.get_value_type(toty))
    else:
        # Unsigned upcast
        return builder.zext(val, context.get_value_type(toty))
