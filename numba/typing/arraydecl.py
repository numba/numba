from __future__ import print_function, division, absolute_import

import itertools

from numba import types, intrinsics
from numba.utils import PYVERSION
from .builtins import normalize_nd_index
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, builtin_global, builtin,
                                    builtin_attr, signature, bound_function,
                                    make_callable_template)


def get_array_index_type(ary, idx):
    """
    Returns None or a tuple-3 for the types of the input array, index, and
    resulting type of ``array[index]``.

    Note: This is shared logic for ndarray getitem and setitem.
    """
    if not isinstance(ary, types.Buffer):
        return

    idx = normalize_nd_index(idx)
    if idx is None:
        return

    if idx == types.slice3_type:
        res = ary.copy(layout='A')
    elif isinstance(idx, (types.UniTuple, types.Tuple)):
        if ary.ndim > len(idx):
            return
        elif ary.ndim < len(idx):
            return
        elif any(i == types.slice3_type for i in idx):
            ndim = ary.ndim
            for i in idx:
                if i != types.slice3_type:
                    ndim -= 1
            res = ary.copy(ndim=ndim, layout='A')
        else:
            res = ary.dtype
    elif isinstance(idx, types.Integer):
        if ary.ndim == 1:
            res = ary.dtype
        elif not ary.slice_is_copy and ary.ndim > 1:
            # Left-index into a F-contiguous array gives a non-contiguous view
            layout = 'C' if ary.layout == 'C' else 'A'
            res = ary.copy(ndim=ary.ndim - 1, layout=layout)
        else:
            return

    else:
        raise Exception("unreachable: index type of %s" % idx)

    if isinstance(res, types.Buffer) and res.slice_is_copy:
        # Avoid view semantics when the original type creates a copy
        # when slicing.
        return

    return ary, idx, res


@builtin
class GetItemBuffer(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        out = get_array_index_type(ary, idx)
        if out is not None:
            ary, idx, res = out
            return signature(res, ary, idx)

@builtin
class SetItemBuffer(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args
        if isinstance(ary, types.Buffer):
            if not ary.mutable:
                raise TypeError("Cannot modify value of type %s" %(ary,))

            out = get_array_index_type(ary, idx)
            if out is not None:
                ary, idx, res = out
                if isinstance(res, types.Array):
                    if res.ndim > 1:
                        raise NotImplementedError(
                            "Cannot store slice on array of more than one dimension")
                    # Allow for broadcasting.
                    if not isinstance(val, types.Array):
                        res = res.dtype
                return signature(types.none, ary, idx, res)


def normalize_shape(shape):
    if isinstance(shape, types.UniTuple):
        if isinstance(shape.dtype, types.Integer):
            dimtype = types.intp if shape.dtype.signed else types.uintp
            return types.UniTuple(dimtype, len(shape))

    elif isinstance(shape, types.Tuple) and shape.count == 0:
        return shape


@builtin_attr
class ArrayAttribute(AttributeTemplate):
    key = types.Array

    def resolve_dtype(self, ary):
        return types.DType(ary.dtype)

    def resolve_itemsize(self, ary):
        return types.intp

    def resolve_shape(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_strides(self, ary):
        return types.UniTuple(types.intp, ary.ndim)

    def resolve_ndim(self, ary):
        return types.intp

    def resolve_size(self, ary):
        return types.intp

    def resolve_flat(self, ary):
        return types.NumpyFlatType(ary)

    def resolve_ctypes(self, ary):
        return types.ArrayCTypes(ary)

    def resolve_flags(self, ary):
        return types.ArrayFlags(ary)

    def resolve_T(self, ary):
        if ary.ndim <= 1:
            retty = ary
        else:
            layout = {"C": "F", "F": "C"}.get(ary.layout, "A")
            retty = ary.copy(layout=layout)
        return retty

    @bound_function("array.transpose")
    def resolve_transpose(self, ary, args, kws):
        assert not args
        assert not kws
        return signature(self.resolve_T(ary))

    @bound_function("array.copy")
    def resolve_copy(self, ary, args, kws):
        assert not args
        assert not kws
        retty = ary.copy(layout="C")
        return signature(retty)

    @bound_function("array.nonzero")
    def resolve_nonzero(self, ary, args, kws):
        assert not args
        assert not kws
        # 0-dim arrays return one result array
        ndim = max(ary.ndim, 1)
        retty = types.UniTuple(types.Array(types.intp, 1, 'C'), ndim)
        return signature(retty)

    @bound_function("array.reshape")
    def resolve_reshape(self, ary, args, kws):
        assert not kws
        shape, = args
        shape = normalize_shape(shape)
        if shape is None:
            return
        if ary.layout == "C":
            # Given order='C' (the only supported value), a C-contiguous
            # array is always returned for a C-contiguous input.
            layout = "C"
        else:
            layout = "A"
        ndim = shape.count if isinstance(shape, types.BaseTuple) else 1
        retty = ary.copy(ndim=ndim)
        return signature(retty, shape)

    @bound_function("array.sort")
    def resolve_sort(self, ary, args, kws):
        assert not args
        assert not kws
        if ary.ndim == 1:
            return signature(types.none)

    @bound_function("array.view")
    def resolve_view(self, ary, args, kws):
        from .npydecl import _parse_dtype
        assert not kws
        dtype, = args
        dtype = _parse_dtype(dtype)
        retty = ary.copy(dtype=dtype)
        return signature(retty, *args)

    def generic_resolve(self, ary, attr):
        # Resolution of other attributes, for record arrays
        if isinstance(ary.dtype, types.Record):
            if attr in ary.dtype.fields:
                return ary.copy(dtype=ary.dtype.typeof(attr), layout='A')


@builtin_attr
class ArrayCTypesAttribute(AttributeTemplate):
    key = types.ArrayCTypes

    def resolve_data(self, ctinfo):
        return types.uintp


@builtin_attr
class ArrayFlagsAttribute(AttributeTemplate):
    key = types.ArrayFlags

    def resolve_contiguous(self, ctflags):
        return types.boolean

    def resolve_c_contiguous(self, ctflags):
        return types.boolean

    def resolve_f_contiguous(self, ctflags):
        return types.boolean


@builtin_attr
class NestedArrayAttribute(ArrayAttribute):
    key = types.NestedArray


def _expand_integer(ty):
    """
    If *ty* is an integer, expand it to a machine int (like Numpy).
    """
    if isinstance(ty, types.Integer):
        if ty.signed:
            return max(types.intp, ty)
        else:
            return max(types.uintp, ty)
    else:
        return ty

def generic_homog(self, args, kws):
    assert not args
    assert not kws
    return signature(self.this.dtype, recvr=self.this)

def generic_expand(self, args, kws):
    return signature(_expand_integer(self.this.dtype), recvr=self.this)

def generic_expand_cumulative(self, args, kws):
    assert isinstance(self.this, types.Array)
    return_type = types.Array(dtype=_expand_integer(self.this.dtype),
                              ndim=1, layout='C')
    return signature(return_type, recvr=self.this)

def generic_hetero_real(self, args, kws):
    assert not args
    assert not kws
    if self.this.dtype in types.integer_domain:
        return signature(types.float64, recvr=self.this)
    return signature(self.this.dtype, recvr=self.this)

def generic_index(self, args, kws):
    assert not args
    assert not kws
    return signature(types.intp, recvr=self.this)

def install_array_method(name, generic):
    my_attr = {"key": "array." + name, "generic": generic}
    temp_class = type("Array_" + name, (AbstractTemplate,), my_attr)

    def array_attribute_attachment(self, ary):
        return types.BoundFunction(temp_class, ary)

    setattr(ArrayAttribute, "resolve_" + name, array_attribute_attachment)

# Functions that return the same type as the array
for fname in ["min", "max"]:
    install_array_method(fname, generic_homog)

# Functions that return a machine-width type, to avoid overflows
for fname in ["sum", "prod"]:
    install_array_method(fname, generic_expand)

# Functions that return a machine-width type, to avoid overflows
for fname in ["cumsum", "cumprod"]:
    install_array_method(fname, generic_expand_cumulative)

# Functions that require integer arrays get promoted to float64 return
for fName in ["mean", "median", "var", "std"]:
    install_array_method(fName, generic_hetero_real)

# Functions that return an index (intp)
install_array_method("argmin", generic_index)
install_array_method("argmax", generic_index)


@builtin
class CmpOpEqArray(AbstractTemplate):
    key = '=='

    def generic(self, args, kws):
        assert not kws
        [va, vb] = args
        if isinstance(va, types.Array) and va == vb:
            return signature(va.copy(dtype=types.boolean), va, vb)
