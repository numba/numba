from __future__ import print_function, division, absolute_import

from collections import namedtuple

from numba import types
from numba.typing.templates import (AttributeTemplate, AbstractTemplate,
                                    builtin, builtin_attr, signature,
                                    bound_function)

Indexing = namedtuple("Indexing", ("index", "result", "advanced"))


def get_array_index_type(ary, idx):
    """
    Returns None or a tuple-3 for the types of the input array, index, and
    resulting type of ``array[index]``.

    Note: This is shared logic for ndarray getitem and setitem.
    """
    if not isinstance(ary, types.Buffer):
        return

    ndim = ary.ndim

    left_indices = []
    right_indices = []
    ellipsis_met = False
    advanced = False
    has_integer = False

    if not isinstance(idx, types.BaseTuple):
        idx = [idx]

    # Walk indices
    for ty in idx:
        if ty is types.ellipsis:
            if ellipsis_met:
                raise TypeError("only one ellipsis allowed in array index "
                                "(got %s)" % (idx,))
            ellipsis_met = True
        elif ty is types.slice3_type:
            pass
        elif isinstance(ty, types.Integer):
            # Normalize integer index
            ty = types.intp if ty.signed else types.uintp
            # Integer indexing removes the given dimension
            ndim -= 1
            has_integer = True
        elif (isinstance(ty, types.Array)
              and ty.ndim == 1
              and isinstance(ty.dtype, (types.Integer, types.Boolean))):
            if advanced or has_integer:
                # We don't support the complicated combination of
                # advanced indices (and integers are considered part
                # of them by Numpy).
                raise NotImplementedError("only one advanced index supported")
            advanced = True
        else:
            raise TypeError("unsupported array index type %s in %s"
                            % (ty, idx))
        (right_indices if ellipsis_met else left_indices).append(ty)

    # Only Numpy arrays support advanced indexing
    if advanced and not isinstance(ary, types.Array):
        return

    # Check indices and result dimensionality
    all_indices = left_indices + right_indices
    n_indices = len(all_indices) - ellipsis_met
    if n_indices > ary.ndim:
        raise TypeError("cannot index %s with %d indices: %s"
                        % (ary, n_indices, idx))
    if (n_indices == ary.ndim
        and all(isinstance(ty, types.Integer) for ty in all_indices)
        and not ellipsis_met):
        # Full integer indexing => scalar result
        # (note if ellipsis is present, a 0-d view is returned instead)
        res = ary.dtype

    elif advanced:
        # Result is a copy
        res = ary.copy(ndim=ndim, layout='C', readonly=False)

    else:
        # Result is a view
        if ary.slice_is_copy:
            # Avoid view semantics when the original type creates a copy
            # when slicing.
            return

        # Infer layout
        layout = ary.layout
        if layout == 'C':
            # Integer indexing on the left keeps the array C-contiguous
            if n_indices == ary.ndim:
                # If all indices are there, ellipsis's place is indifferent
                left_indices = left_indices + right_indices
                right_indices = []
            if right_indices:
                layout = 'A'
            else:
                for ty in left_indices:
                    if ty is not types.ellipsis and not isinstance(ty, types.Integer):
                        # Slicing cannot guarantee to keep the array contiguous
                        layout = 'A'
                        break
        elif layout == 'F':
            # Integer indexing on the right keeps the array F-contiguous
            if n_indices == ary.ndim:
                # If all indices are there, ellipsis's place is indifferent
                right_indices = left_indices + right_indices
                left_indices = []
            if left_indices:
                layout = 'A'
            else:
                for ty in right_indices:
                    if ty is not types.ellipsis and not isinstance(ty, types.Integer):
                        # Slicing cannot guarantee to keep the array contiguous
                        layout = 'A'
                        break

        res = ary.copy(ndim=ndim, layout=layout)

    # Re-wrap indices
    if isinstance(idx, types.BaseTuple):
        idx = types.BaseTuple.from_types(all_indices)
    else:
        idx, = all_indices

    return Indexing(idx, res, advanced)


@builtin
class GetItemBuffer(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        out = get_array_index_type(ary, idx)
        if out is not None:
            return signature(out.result, ary, out.index)

@builtin
class SetItemBuffer(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args
        if not isinstance(ary, types.Buffer):
            return
        if not ary.mutable:
            raise TypeError("Cannot modify value of type %s" %(ary,))
        out = get_array_index_type(ary, idx)
        if out is None:
            return

        idx = out.index
        res = out.result
        if isinstance(res, types.Array):
            # Indexing produces an array
            if not isinstance(val, types.Array):
                # Allow scalar broadcasting
                res = res.dtype
            elif (val.ndim == res.ndim and
                  self.context.can_convert(val.dtype, res.dtype)):
                # Allow assignement of same-dimensionality compatible-dtype array
                res = val
            else:
                # Unexpected dimensionality of assignment source
                # (array broadcasting is unsupported)
                return
        elif not isinstance(val, types.Array):
            # Single item assignment
            res = val
        else:
            return
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
        def sentry_shape_scalar(ty):
            if ty in types.number_domain:
                # Guard against non integer type
                if not isinstance(ty, types.Integer):
                    raise TypeError("reshape() arg cannot be {0}".format(ty))
                return True
            else:
                return False

        assert not kws
        if ary.layout not in 'CF':
            # only work for contiguous array
            raise TypeError("reshape() supports contiguous array only")

        if len(args) == 1:
            # single arg
            shape, = args

            if sentry_shape_scalar(shape):
                ndim = 1
            else:
                shape = normalize_shape(shape)
                if shape is None:
                    return
                ndim = shape.count
            retty = ary.copy(ndim=ndim)
            return signature(retty, shape)

        elif len(args) == 0:
            # no arg
            raise TypeError("reshape() take at least one arg")

        else:
            # vararg case
            if any(not sentry_shape_scalar(a) for a in args):
                raise TypeError("reshape({0}) is not supported".format(
                    ', '.join(args)))

            retty = ary.copy(ndim=len(args))
            return signature(retty, *args)

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


@builtin
class StaticGetItemArray(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        # Resolution of members for record and structured arrays
        ary, idx = args
        if (isinstance(ary, types.Array) and isinstance(idx, str) and
            isinstance(ary.dtype, types.Record)):
            if idx in ary.dtype.fields:
                return ary.copy(dtype=ary.dtype.typeof(idx), layout='A')


@builtin_attr
class RecordAttribute(AttributeTemplate):
    key = types.Record

    def generic_resolve(self, record, attr):
        ret = record.typeof(attr)
        assert ret
        return ret

@builtin
class StaticGetItemRecord(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        # Resolution of members for records
        record, idx = args
        if isinstance(record, types.Record) and isinstance(idx, str):
            ret = record.typeof(idx)
            assert ret
            return ret

@builtin
class StaticSetItemRecord(AbstractTemplate):
    key = "static_setitem"

    def generic(self, args, kws):
        # Resolution of members for record and structured arrays
        record, idx, value = args
        if isinstance(record, types.Record) and isinstance(idx, str):
            expectedty = record.typeof(idx)
            if self.context.can_convert(value, expectedty) is not None:
                return signature(types.void, record, types.Const(idx), value)


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
