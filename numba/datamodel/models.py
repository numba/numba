from __future__ import print_function, absolute_import

from llvmlite import ir
from numba import types, numpy_support
from .registry import register_default


class DataModel(object):
    """
    DataModel describe how a FE type is represented in the LLVM IR at
    different contexts.

    Contexts are:

    - value: representation inside function body.  Maybe stored in stack.
    The representation here are flexible.

    - data: representation used when storing into containers (e.g. arrays).

    - argument: representation used for function argument.  All composite
    types are unflattened into multiple primitive types.

    - return: representation used for return argument.

    Throughput the compiler pipeline, a LLVM value is usually passed around
    in the "value" representation.  All "as_" prefix function converts from
    "value" representation.  All "from_" prefix function converts to the
    "value"  representation.

    """
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
        """Return a LLVM type or nested tuple of LLVM type
        """
        return self.get_value_type()

    def get_return_type(self):
        return self.get_value_type()

    def as_data(self, builder, value):
        return NotImplemented

    def as_argument(self, builder, value):
        """
        Takes one LLVM value
        Return a LLVM value or nested tuple of LLVM value
        """
        return NotImplemented

    def as_return(self, builder, value):
        return NotImplemented

    def from_data(self, builder, value):
        return NotImplemented

    def from_argument(self, builder, value):
        """
        Takes a LLVM value or nested tuple of LLVM value
        Returns one LLVM value
        """
        return NotImplemented

    def from_return(self, builder, value):
        return NotImplemented

    def load_from_data_pointer(self, builder, value):
        """Only the record model this for pass by reference semantic.
        """
        return NotImplemented

    def _compared_fields(self):
        return (type(self), self._fe_type)

    def __hash__(self):
        return hash(tuple(self._compared_fields()))

    def __eq__(self, other):
        if type(self) is type(other):
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
@register_default(types.Module)
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


@register_default(types.Float)
class FloatModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        if fe_type == types.float32:
            be_type = ir.FloatType()
        elif fe_type == types.float64:
            be_type = ir.DoubleType()
        else:
            raise NotImplementedError(fe_type)
        super(FloatModel, self).__init__(dmm, fe_type, be_type)


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
        return (self._elem_model.get_argument_type(),) * self._count

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
        """
        Converts the LLVM struct in `value` into a representation suited for
        storing into arrays.

        Note
        ----
        Current implementation rarely changes how types are represented for
        "value" and "data".  This is usually a pointless rebuild of the
        immutable LLVM struct value.  Luckily, LLVM optimization removes all
        redundancy.

        Sample usecase: Structures nested with pointers to other structures
        that can be serialized into  a flat representation when storing into
        array.
        """
        elems = self._as("as_data", builder, value)
        struct = ir.Constant(self.get_data_type(), ir.Undefined)
        for i, el in enumerate(elems):
            struct = builder.insert_value(struct, el, [i])
        return struct

    def from_data(self, builder, value):
        """
        Convert from "data" representation back into "value" representation.
        Usually invoked when loading from array.

        See notes in `as_data()`
        """
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
        """Get a field at the given position or the fieldname

        Args
        ----
        builder:
            LLVM IRBuilder
        val:
            value to be inserted
        pos: int or str
            field index or field name

        Returns
        -------
        Extracted value
        """
        if isinstance(pos, str):
            pos = self.get_field_position(pos)
        return builder.extract_value(val, [pos],
                                     name="extracted." + self._fields[pos])

    def set(self, builder, stval, val, pos):
        """Set a field at the given position or the fieldname

        Args
        ----
        builder:
            LLVM IRBuilder
        stval:
            LLVM struct value
        val:
            value to be inserted
        pos: int or str
            field index or field name

        Returns
        -------
        A new LLVM struct with the value inserted
        """
        if isinstance(pos, str):
            pos = self.get_field_position(pos)
        return builder.insert_value(stval, val, [pos],
                                    name="inserted." + self._fields[pos])

    def get_field_position(self, field):
        return self._fields.index(field)

    @property
    def field_count(self):
        return len(self._fields)

    def get_type(self, pos):
        """Get the frontend type (numba type) of a field given the position
         or the fieldname

        Args
        ----
        pos: int or str
            field index or field name

        """
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


@register_default(types.Pair)
class PairModel(StructModel):
    def __init__(self, dmm, fe_type):
        members = [('first', fe_type.first_type),
                   ('second', fe_type.second_type)]
        super(PairModel, self).__init__(dmm, fe_type, members)


@register_default(types.Array)
@register_default(types.NestedArray)
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


class CContiguousFlatIter(StructModel):
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
        super(CContiguousFlatIter, self).__init__(dmm, fe_type, members)


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


@register_default(types.ZipType)
class ZipType(StructModel):
    def __init__(self, dmm, fe_type):
        members = [('iter%d' % i, source_type.iterator_type)
                   for i, source_type in enumerate(fe_type.source_types)]
        super(ZipType, self).__init__(dmm, fe_type, members)


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


@register_default(types.NumpyFlatType)
def handle_numpy_flat_type(dmm, ty):
    if ty.array_type.layout == 'C':
        return CContiguousFlatIter(dmm, ty)
    else:
        return FlatIter(dmm, ty)


@register_default(types.Structure)
def handle_structure(dmm, ty):
    return StructModel(dmm, ty, ty.members)

@register_default(types.StructRef)
def handle_structref(dmm, ty):
    return dmm[types.CPointer(ty.base)]
