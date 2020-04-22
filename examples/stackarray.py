from numba.core.extending import type_callable, \
    models, register_model, make_attribute_wrapper, lower_builtin, \
    box, overload
from numba.core import cgutils
from numba import types, njit
import operator


# Array
class StackArray(object):

    def __init__(self, size, ptr):
        self.size = size
        self.ptr = ptr

    def __str__(self):
        return "StackArray(%d)" % self.size

    def __repr__(self):
        return self.__str__()


# Typing layer
## Creating a new Numba type
class StackArrayType(types.Type):
    def __init__(self, dtype):
        self.dtype = dtype
        name = "StackArrayType<%s>" % dtype
        super(StackArrayType, self).__init__(name=name)

    @property
    def key(self):
        return self.dtype


## Type inference for operations
@type_callable(StackArray)
def type_stackarray(context):
    def typer(size, array_type):
        return StackArrayType(array_type.dtype)
    return typer


# Extending the lowering model
## Model
@register_model(StackArrayType)
class StackArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # fe_type is StackArrayType
        members = [
            ('size', types.int64),
            ('ptr', types.CPointer(fe_type.key)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


## Exposing attributes
make_attribute_wrapper(StackArrayType, 'size', 'size')
make_attribute_wrapper(StackArrayType, 'ptr', 'ptr')


## Implementing the constructor
@lower_builtin(StackArray, types.IntegerLiteral, types.Any)
def impl_stackarray(context, builder, sig, args):

    typ = sig.return_type
    [size, array_type] = args
    fa = cgutils.create_struct_proxy(typ)(context, builder)
    lty = context.get_value_type(sig.return_type.key)
    fa.size = size
    fa.ptr = cgutils.alloca_once(builder, lty,
                                 size=sig.args[0].literal_value)  # stack alloc
    return fa._getvalue()


@overload(operator.setitem)
def stackarray_setitem(a, i, v):
    if isinstance(a, StackArrayType):
        def impl(a, i, v):
            a.ptr[i] = v
        return impl


@overload(operator.getitem)
def stackarray_getitem(a, i):
    if isinstance(a, StackArrayType):
        def impl(a, i):
            return a.ptr[i]
        return impl


@overload(len)
def stack_array_len(x):
    if isinstance(x, StackArrayType):
        def impl(x):
            return x.size
        return impl


# Boxing
@box(StackArrayType)
def box_fixedlength(typ, val, c):
    """
    Convert a native fixed array structure to an StackArray object.
    """
    context = c.context
    builder = c.builder
    pyapi = c.pyapi

    stackarray = cgutils.create_struct_proxy(typ)(context, builder, value=val)
    size_obj = pyapi.long_from_long(stackarray.size)
    ptr_obj = pyapi.unserialize(pyapi.serialize_object(stackarray.ptr))
    class_obj = pyapi.unserialize(pyapi.serialize_object(StackArray))
    res = pyapi.call_function_objargs(class_obj, (size_obj, ptr_obj))
    pyapi.decref(size_obj)
    pyapi.decref(class_obj)
    return res


@njit
def foo():
    f = StackArray(4, types.int64)
    for i in range(len(f)):
        f[i] = i
    s = 0
    for i in range(len(f)):
        s += f[i]
    return s, f


s, f = foo()
print(s, f, sep=', ') # prints "6, StackArray(4)"
