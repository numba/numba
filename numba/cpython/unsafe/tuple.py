"""
This file provides internal compiler utilities that support certain special
operations with tuple and workarounds for limitations enforced in userland.
"""
import inspect

from numba.core import types, typing
from numba.core.errors import RequireLiteralValue
from numba.core.extending import intrinsic


# QUESTION: Is there a good place for this function somewhere else?
#           I keep writing boiler-plate type checks and having a function that
#           does this for most applications would get rid of much of the
#           boiler-plate.
def validate_arg_type(arg, arg_name, valid_types, none_allowed=False,
                      msg_prefix=None):
    """Checks that a parameter of an overload has the correct type and prints
    appropriate error message if not.

    Usage: validate_arg_type(tup, 'tup', types.BaseTuple,
                             msg_prefix='tuple_setitem: ')
    """
    if not isinstance(valid_types, tuple):
        valid_types = (valid_types, )

    if msg_prefix is None:
        msg_prefix = ''

    # determine the type name to show to the user
    name_mapping = {
        types.Boolean: 'boolean',
        types.BaseTuple: 'tuple',
        types.IntegerLiteral: 'constant (literal) integer'
    }

    def name_for_type(type_):
        # any defined name mapping takes precedence
        name = name_mapping.get(type_)
        if name is not None:
            return name
        # if a type instance is defined, it's name takes precendence
        name = getattr(type_, 'name', None)
        if name is not None:
            return name
        # fallback to type name
        name = getattr(type_, '__name__', None)
        if name is not None:
            return name
        raise ValueError(f"Cannot find name for type {type_}")

    valid_names = [name_for_type(type_) for type_ in valid_types]

    if none_allowed is True:
        valid_types = valid_types + (types.NoneType, type(None))

    have_type_class = all(inspect.isclass(x) for x in valid_types)

    def create_error_msg():
        if len(valid_names) == 1:
            msg = (f"{msg_prefix}argument '{arg_name}' must of type "
                   f"{valid_names[0]}, got type {type(arg).__name__}")
        else:
            msg = (f"{msg_prefix}argument '{arg_name}' must of the "
                   f"following types: {', '.join(valid_names)}; got type "
                   f"{type(arg).__name__}")
        return msg

    if have_type_class:
        # have type class
        if not isinstance(arg, valid_types):
            msg = create_error_msg()
            if all(issubclass(x, types.Literal) for x in valid_types):
                raise RequireLiteralValue(msg)
            else:
                raise TypeError(msg)
    else:
        # have instance of type class
        if arg not in valid_types:
            msg = create_error_msg()
            if all(isinstance(x, types.Literal) for x in valid_types):
                raise RequireLiteralValue(msg)
            else:
                raise TypeError(msg)


@intrinsic
def tuple_setitem(typingctx, tup, idx, val):
    """Return a copy of the tuple with item at *idx* replaced with *val*.

    Operation: ``out = tup[:idx] + (val,) + tup[idx + 1:]

    **Warning**

    - The dtype of the tuple cannot be changed.
    """
    sig = tup(tup, idx, val)

    validate_arg_type(tup, 'tup', types.BaseTuple,
                      msg_prefix='tuple_setitem: ')
    tuple_length = len(tup)

    # check bounds
    validate_arg_type(idx, 'idx', types.IntegerLiteral,
                      msg_prefix='tuple_setitem: ')
    idx_ = idx.literal_value
    if not (0 <= idx_ < tuple_length):
        raise IndexError('tuple_setitem(): tuple index out of range')

    # check that the type of the new value fits the tuple type
    if isinstance(val, types.Literal):
        val_type = val.literal_type
    else:
        val_type = val
    validate_arg_type(val_type, 'val', tup[idx_],
                      msg_prefix='tuple_setitem: ')

    idx_2 = idx_ + 1

    def create_new_tuple(tup, val):
        return tup[:idx_] + (val,) + tup[idx_2:]

    def codegen(context, builder, signature, args):
        tup_type, idx_type, val_type = signature.args
        tup, idx, val = args

        # create new tuple with replaced value
        inner_argtypes = [tup_type, val_type]
        inner_sig = typing.signature(tup_type, *inner_argtypes)
        new_tup = context.compile_internal(builder, create_new_tuple,
                                           inner_sig, [tup, val])

        return new_tup

    return sig, codegen
