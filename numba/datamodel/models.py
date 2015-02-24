from __future__ import print_function, absolute_import

from llvmlite import ir
from numba import types, numpy_support
from .registry import register_default


class DataModel(object):
    def __init__(self, dmm, fe_type):
        self._dmm = dmm
        self._fe_type = fe_type

    @property
    def fe_type(self):
        return self._fe_type

    def get_value_type(self):
        raise NotImplementedError

    def get_data_type(self):
        return self.get_value_type()

    def get_argument_type(self):
        return self.get_value_type()

    def get_return_type(self):
        return self.get_value_type()

    def as_data(self, builder, value):
        return NotImplemented

    def as_argument(self, builder, value):
        return NotImplemented

    def as_return(self, builder, value):
        return NotImplemented

    def from_data(self, builder, value):
        return NotImplemented

    def from_argument(self, builder, value):
        return NotImplemented

    def from_return(self, builder, value):
        return NotImplemented

    def load_from_data_pointer(self, builder, value):
        return NotImplemented

    def _compared_fields(self):
        return (type(self), self._fe_type)

    def __hash__(self):
        return hash(tuple(self._compared_fields()))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._compared_fields() == other._compared_fields()
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


@register_default(types.Boolean)
class BooleanModel(DataModel):
    def get_value_type(self):
        return ir.IntType(1)

    def get_data_type(self):
        return ir.IntType(8)

    def get_return_type(self):
        return self.get_data_type()

    def get_argument_type(self):
        return self.get_data_type()

    def as_data(self, builder, value):
        return builder.zext(value, self.get_data_type())

    def as_argument(self, builder, value):
        return self.as_data(builder, value)

    def as_return(self, builder, value):
        return self.as_data(builder, value)

    def from_data(self, builder, value):
        return builder.trunc(value, self.get_value_type())

    def from_argument(self, builder, value):
        return self.from_data(builder, value)

    def from_return(self, builder, value):
        return self.from_data(builder, value)


class PrimitiveModel(DataModel):
    """A primitive type can be represented natively in the target in all
    usage contexts.
    """

    def __init__(self, dmm, fe_type, be_type):
        super(PrimitiveModel, self).__init__(dmm, fe_type)
        self.be_type = be_type

    def get_value_type(self):
        return self.be_type

    def as_data(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return self.as_data(builder, value)

    def as_return(self, builder, value):
        return self.as_data(builder, value)

    def from_data(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return self.from_data(builder, value)

    def from_return(self, builder, value):
        return self.from_data(builder, value)


@register_default(types.Opaque)
@register_default(types.NoneType)
@register_default(types.Function)
@register_default(types.Type)
@register_default(types.Object)
class OpaqueModel(PrimitiveModel):
    """
    Passed as opaque pointers
    """

    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(8).as_pointer()
        super(OpaqueModel, self).__init__(dmm, fe_type, be_type)


@register_default(types.Integer)
class IntegerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(fe_type.bitwidth)
        super(IntegerModel, self).__init__(dmm, fe_type, be_type)


class FloatModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super(FloatModel, self).__init__(dmm, fe_type, ir.FloatType())


class DoubleModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super(DoubleModel, self).__init__(dmm, fe_type, ir.DoubleType())


@register_default(types.CPointer)
class PointerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = dmm.lookup(fe_type.dtype).get_data_type().as_pointer()
        super(PointerModel, self).__init__(dmm, fe_type, be_type)


@register_default(types.UniTuple)
class UniTupleModel(DataModel):
    def __init__(self, dmm, fe_type):
        super(UniTupleModel, self).__init__(dmm, fe_type)
        self._elem_model = dmm.lookup(fe_type.dtype)
        self._count = len(fe_type)

    def get_value_type(self):
        elem_type = self._elem_model.get_value_type()
        return ir.ArrayType(elem_type, self._count)

    def get_data_type(self):
        elem_type = self._elem_model.get_data_type()
        return ir.ArrayType(elem_type, self._count)

    def get_return_type(self):
        return self.get_value_type()

    def get_argument_type(self):
        return [self._elem_model.get_argument_type()] * self._count

    def as_argument(self, builder, value):
        out = []
        for i in range(self._count):
            out.append(builder.extract_value(value, [i]))
        return out

    def from_argument(self, builder, value):
        out = ir.Constant(self.get_value_type(), ir.Undefined)
        for i, v in enumerate(value):
            out = builder.insert_value(out, v, [i])
        return out

    def as_data(self, builder, value):
        out = ir.Constant(self.get_data_type(), ir.Undefined)
        for i in range(self._count):
            val = builder.extract_value(value, [i])
            dval = self._elem_model.as_data(builder, val)
            out = builder.insert_value(out, dval, [i])
        return out

    def from_data(self, builder, value):
        out = ir.Constant(self.get_value_type(), ir.Undefined)
        for i in range(self._count):
            val = builder.extract_value(value, [i])
            dval = self._elem_model.from_data(builder, val)
            out = builder.insert_value(out, dval, [i])
        return out

    def as_return(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value


class CompositeModel(DataModel):
    """Any model that is composed of multiple other models should subclass from
    this.
    """
    pass


class StructModel(CompositeModel):
    def __init__(self, dmm, fe_type, members):
        super(StructModel, self).__init__(dmm, fe_type)
        if members:
            self._fields, self._members = zip(*members)
        else:
            self._fields = self._members = ()
        self._models = tuple([self._dmm.lookup(t) for t in self._members])

    def get_value_type(self):
        elems = [t.get_value_type() for t in self._models]
        return ir.LiteralStructType(elems)

    def get_data_type(self):
        elems = [t.get_data_type() for t in self._models]
        return ir.LiteralStructType(elems)

    def get_argument_type(self):
        return tuple([t.get_argument_type() for t in self._models])

    def get_return_type(self):
        return self.get_data_type()

    def _as(self, methname, builder, value):
        extracted = []
        for i, dm in enumerate(self._models):
            extracted.append(getattr(dm, methname)(builder,
                                                   self.get(builder, value, i)))
        return tuple(extracted)

    def _from(self, methname, builder, value):
        struct = ir.Constant(self.get_value_type(), ir.Undefined)

        for i, (dm, val) in enumerate(zip(self._models, value)):
            v = getattr(dm, methname)(builder, val)
            struct = self.set(builder, struct, v, i)

        return struct

    def as_data(self, builder, value):
        elems = self._as("as_data", builder, value)
        struct = ir.Constant(self.get_data_type(), ir.Undefined)
        for i, el in enumerate(elems):
            struct = builder.insert_value(struct, el, [i])
        return struct

    def from_data(self, builder, value):
        vals = [builder.extract_value(value, [i])
                for i in range(len(self._members))]
        return self._from("from_data", builder, vals)

    def as_argument(self, builder, value):
        return self._as("as_argument", builder, value)

    def from_argument(self, builder, value):
        return self._from("from_argument", builder, value)

    def as_return(self, builder, value):
        elems = self._as("as_data", builder, value)
        struct = ir.Constant(self.get_data_type(), ir.Undefined)
        for i, el in enumerate(elems):
            struct = builder.insert_value(struct, el, [i])
        return struct

    def from_return(self, builder, value):
        vals = [builder.extract_value(value, [i])
                for i in range(len(self._members))]
        return self._from("from_data", builder, vals)

    def get(self, builder, val, pos):
        if isinstance(pos, str):
            pos = self.get_field_position(pos)
        return builder.extract_value(val, [pos])

    def set(self, builder, stval, val, pos):
        if isinstance(pos, str):
            pos = self.get_field_position(pos)
        return builder.insert_value(stval, val, [pos])

    def get_field_position(self, field):
        return self._fields.index(field)

    def get_type(self, pos):
        if isinstance(pos, str):
            pos = self.get_field_position(pos)
        return self._members[pos]


@register_default(types.Complex)
class ComplexModel(StructModel):
    _element_type = NotImplemented

    def __init__(self, dmm, fe_type):
        members = [
            ('real', fe_type.underlying_float),
            ('imag', fe_type.underlying_float),
        ]
        super(ComplexModel, self).__init__(dmm, fe_type, members)


@register_default(types.Tuple)
class TupleModel(StructModel):
    def __init__(self, dmm, fe_type):
        members = [('f' + str(i), t) for i, t in enumerate(fe_type)]
        super(TupleModel, self).__init__(dmm, fe_type, members)


@register_default(types.Array)
class ArrayModel(StructModel):
    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            ('parent', types.pyobject),
            ('nitems', types.intp),
            ('itemsize', types.intp),
            ('data', types.CPointer(fe_type.dtype)),
            ('shape', types.UniTuple(types.intp, ndim)),
            ('strides', types.UniTuple(types.intp, ndim)),
        ]
        super(ArrayModel, self).__init__(dmm, fe_type, members)

    def as_data(self, builder, value):
        return NotImplemented

    def from_data(self, builder, value):
        return NotImplemented


@register_default(types.Optional)
class OptionalModel(StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.type),
            ('valid', types.boolean),
        ]
        self._value_model = dmm.lookup(fe_type.type)
        super(OptionalModel, self).__init__(dmm, fe_type, members)

    def get_return_type(self):
        return self._value_model.get_return_type()

    def as_return(self, builder, value):
        return NotImplemented

    def from_return(self, builder, value):
        return self._value_model.from_return(builder, value)


@register_default(types.Record)
class RecordModel(CompositeModel):
    def __init__(self, dmm, fe_type):
        super(RecordModel, self).__init__(dmm, fe_type)
        self._models = [self._dmm.lookup(t) for _, t in fe_type.members]
        self._be_type = ir.ArrayType(ir.IntType(8), fe_type.size)

    def get_value_type(self):
        """Passed around as reference to underlying data
        """
        return self._be_type.as_pointer()

    def get_argument_type(self):
        return self.get_value_type()

    def get_return_type(self):
        return self.get_value_type()

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

    def load_from_data_pointer(self, builder, ptr):
        return builder.bitcast(ptr, self.get_value_type())


@register_default(types.UnicodeCharSeq)
class UnicodeCharSeq(DataModel):
    def __init__(self, dmm, fe_type):
        super(UnicodeCharSeq, self).__init__(dmm, fe_type)
        charty = ir.IntType(numpy_support.sizeof_unicode_char * 8)
        self._be_type = ir.ArrayType(charty, fe_type.count)

    def get_value_type(self):
        return self._be_type

    def get_data_type(self):
        return self._be_type


@register_default(types.CharSeq)
class CharSeq(DataModel):
    def __init__(self, dmm, fe_type):
        super(CharSeq, self).__init__(dmm, fe_type)
        charty = ir.IntType(8)
        self._be_type = ir.ArrayType(charty, fe_type.count)

    def get_value_type(self):
        return self._be_type

    def get_data_type(self):
        return self._be_type

    def as_data(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value


class CConitugousFlatIter(StructModel):
    def __init__(self, dmm, fe_type):
        assert fe_type.array_type.layout == 'C'
        array_type = fe_type.array_type
        dtype = array_type.dtype
        members = [('array', types.CPointer(array_type)),
                   ('stride', types.intp),
                   ('pointer', types.CPointer(types.CPointer(dtype))),
                   ('index', types.CPointer(types.intp)),
                   ('indices', types.CPointer(types.intp)),
        ]
        super(CConitugousFlatIter, self).__init__(dmm, fe_type, members)


class FlatIter(StructModel):
    def __init__(self, dmm, fe_type):
        array_type = fe_type.array_type
        dtype = array_type.dtype
        members = [('array', types.CPointer(array_type)),
                   ('pointers', types.CPointer(types.CPointer(dtype))),
                   ('indices', types.CPointer(types.intp)),
                   ('exhausted', types.CPointer(types.boolean)),
        ]
        super(FlatIter, self).__init__(dmm, fe_type, members)


@register_default(types.UniTupleIter)
class UniTupleIter(StructModel):
    def __init__(self, dmm, fe_type):
        members = [('index', types.CPointer(types.intp)),
                   ('tuple', fe_type.unituple,)]
        super(UniTupleIter, self).__init__(dmm, fe_type, members)


@register_default(types.Slice3Type)
class Slice3(StructModel):
    def __init__(self, dmm, fe_type):
        members = [('start', types.intp),
                   ('stop', types.intp),
                   ('step', types.intp)]
        super(Slice3, self).__init__(dmm, fe_type, members)


@register_default(types.NPDatetime)
@register_default(types.NPTimedelta)
class NPDatetimeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(64)
        super(NPDatetimeModel, self).__init__(dmm, fe_type, be_type)


@register_default(types.ArrayIterator)
class ArrayIterator(StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [('index', types.CPointer(types.uintp)),
                   ('array', fe_type.array_type)]
        super(ArrayIterator, self).__init__(dmm, fe_type, members)


@register_default(types.EnumerateType)
class EnumerateType(StructModel):
    def __init__(self, dmm, fe_type):
        members = [('count', types.CPointer(types.intp)),
                   ('iter', fe_type.source_type)]

        super(EnumerateType, self).__init__(dmm, fe_type, members)


@register_default(types.RangeIteratorType)
class RangeIteratorType(StructModel):
    def __init__(self, dmm, fe_type):
        int_type = fe_type.yield_type
        members = [('iter', types.CPointer(int_type)),
                   ('stop', int_type),
                   ('step', int_type),
                   ('count', types.CPointer(int_type))]
        super(RangeIteratorType, self).__init__(dmm, fe_type, members)


# =============================================================================


@register_default(types.Float)
def handle_floats(dmm, ty):
    if ty == types.float32:
        return FloatModel(dmm, ty)
    elif ty == types.float64:
        return DoubleModel(dmm, ty)
    else:
        raise NotImplementedError(ty)


@register_default(types.NumpyFlatType)
def handle_numpy_flat_type(dmm, ty):
    if ty.array_type.layout == 'C':
        return CConitugousFlatIter(dmm, ty)
    else:
        return FlatIter(dmm, ty)


