from numba import ir, macro, types

def new(shape, dtype):
    from numba import typing
    ndim = 1
    if isinstance(shape, tuple):
        ndim = len(shape)
    elif not isinstance(shape, int):
        raise TypeError("invalid type for shape; got {0}".format(type(shape)))
    restype = types.Array(dtype, ndim, 'C')
    if isinstance(shape, int):
        sig = typing.signature(restype, types.intp, types.Any)
    else:
        sig = typing.signature(restype, types.UniTuple(types.intp, ndim),
                               types.Any)
    return ir.StackArray(sig, shape, dtype)

new = macro.Macro('stack_array.new', new, callable=True,
                  argnames=['shape', 'dtype'])
