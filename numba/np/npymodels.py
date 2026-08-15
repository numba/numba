from llvmlite import ir

from numba.core.datamodel.registry import register_default
from numba.core import types
from numba.core.datamodel.models import (
    CompositeModel, StructModel, ArrayModel, OpaqueModel,
    CContiguousFlatIter, FlatIter
)


@register_default(types.Record)
class RecordModel(CompositeModel):
    def __init__(self, dmm, fe_type):
        super(RecordModel, self).__init__(dmm, fe_type)
        self._models = [self._dmm.lookup(t) for _, t in fe_type.members]
        self._be_type = ir.ArrayType(ir.IntType(8), fe_type.size)
        self._be_ptr_type = self._be_type.as_pointer()

    def get_value_type(self):
        """Passed around as reference to underlying data
        """
        return self._be_ptr_type

    def get_argument_type(self):
        return self._be_ptr_type

    def get_return_type(self):
        return self._be_ptr_type

    def get_data_type(self):
        return self._be_type

    def as_data(self, builder, value):
        return builder.load(value)

    def from_data(self, builder, value):
        raise NotImplementedError("use load_from_data_pointer() instead")

    def as_argument(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value

    def load_from_data_pointer(self, builder, ptr, align=None):
        return builder.bitcast(ptr, self.get_value_type())


@register_default(types.NumpyNdIndexType)
class NdIndexModel(StructModel):
    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [('shape', types.UniTuple(types.intp, ndim)),
                   ('indices', types.EphemeralArray(types.intp, ndim)),
                   ('exhausted', types.EphemeralPointer(types.boolean)),
                   ]
        super(NdIndexModel, self).__init__(dmm, fe_type, members)


@register_default(types.NumpyFlatType)
def handle_numpy_flat_type(dmm, ty):
    if ty.array_type.layout == 'C':
        return CContiguousFlatIter(dmm, ty, need_indices=False)
    else:
        return FlatIter(dmm, ty)


@register_default(types.NumpyNdEnumerateType)
def handle_numpy_ndenumerate_type(dmm, ty):
    if ty.array_type.layout == 'C':
        return CContiguousFlatIter(dmm, ty, need_indices=True)
    else:
        return FlatIter(dmm, ty)


@register_default(types.NumpyNdIterType)
class NdIter(StructModel):
    def __init__(self, dmm, fe_type):
        array_types = fe_type.arrays
        ndim = fe_type.ndim
        shape_len = ndim if fe_type.need_shaped_indexing else 1
        members = [('exhausted', types.EphemeralPointer(types.boolean)),
                   ('arrays', types.Tuple(array_types)),
                   # The iterator's main shape and indices
                   ('shape', types.UniTuple(types.intp, shape_len)),
                   ('indices', types.EphemeralArray(types.intp, shape_len)),
                   ]
        # Indexing state for the various sub-iterators
        # XXX use a tuple instead?
        for i, sub in enumerate(fe_type.indexers):
            kind, start_dim, end_dim, _ = sub
            member_name = 'index%d' % i
            if kind == 'flat':
                # A single index into the flattened array
                members.append(
                    (member_name, types.EphemeralPointer(types.intp))
                )
            elif kind in ('scalar', 'indexed', '0d'):
                # Nothing required
                pass
            else:
                assert 0
        # Slots holding values of the scalar args
        # XXX use a tuple instead?
        for i, ty in enumerate(fe_type.arrays):
            if not isinstance(ty, types.Array):
                member_name = 'scalar%d' % i
                members.append((member_name, types.EphemeralPointer(ty)))

        super(NdIter, self).__init__(dmm, fe_type, members)


@register_default(types.ArrayCTypes)
class ArrayCTypesModel(StructModel):
    def __init__(self, dmm, fe_type):
        # ndim = fe_type.ndim
        members = [('data', types.CPointer(fe_type.dtype)),
                   ('meminfo', types.MemInfoPointer(fe_type.dtype))]
        super(ArrayCTypesModel, self).__init__(dmm, fe_type, members)


@register_default(types.ArrayFlags)
class ArrayFlagsModel(StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('parent', fe_type.array_type),
        ]
        super(ArrayFlagsModel, self).__init__(dmm, fe_type, members)


@register_default(types.NestedArray)
class NestedArrayModel(ArrayModel):
    def __init__(self, dmm, fe_type):
        self._be_type = dmm.lookup(fe_type.dtype).get_data_type()
        super(NestedArrayModel, self).__init__(dmm, fe_type)

    def as_storage_type(self):
        """Return the LLVM type representation for the storage of
        the nestedarray.
        """
        ret = ir.ArrayType(self._be_type, self._fe_type.nitems)
        return ret


register_default(types.Array)(ArrayModel)

register_default(types.DType)(OpaqueModel)
