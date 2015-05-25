from __future__ import print_function, absolute_import

from llvmlite import ir

from numba import cgutils, types, numpy_support
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
        raise NotImplementedError

    def as_argument(self, builder, value):
        """
        Takes one LLVM value
        Return a LLVM value or nested tuple of LLVM value
        """
        raise NotImplementedError(self)

    def as_return(self, builder, value):
        raise NotImplementedError(self)

    def from_data(self, builder, value):
        raise NotImplementedError(self)

    def from_argument(self, builder, value):
        """
        Takes a LLVM value or nested tuple of LLVM value
        Returns one LLVM value
        """
        raise NotImplementedError(self)

    def from_return(self, builder, value):
        raise NotImplementedError

    def load_from_data_pointer(self, builder, ptr):
        """
        Load value from a pointer to data.
        This is the default implementation, sufficient for most purposes.
        """
        return self.from_data(builder, builder.load(ptr))

    def traverse(self, builder, value):
        """
        Traverse contained values
        Returns a iterable of contained (types, values)
        """
        return ()

    def get_nrt_meminfo(self, builder, value):
        """
        Returns the MemInfo object or None if it is not tracked.
        It is only defined for types.meminfo_pointer
        """
        return None


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
        return value

    def as_return(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value


@register_default(types.Opaque)
@register_default(types.NoneType)
@register_default(types.Function)
@register_default(types.Type)
@register_default(types.Object)
@register_default(types.Module)
@register_default(types.Phantom)
@register_default(types.Dispatcher)
@register_default(types.ExceptionType)
@register_default(types.Dummy)
@register_default(types.ExceptionInstance)
@register_default(types.ExternalFunction)
@register_default(types.Method)
@register_default(types.Macro)
@register_default(types.NumberClass)
@register_default(types.DType)
class OpaqueModel(PrimitiveModel):
    """
    Passed as opaque pointers
    """

    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(8).as_pointer()
        super(OpaqueModel, self).__init__(dmm, fe_type, be_type)

    def get_nrt_meminfo(self, builder, value):
        if self._fe_type == types.meminfo_pointer:
            return value


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
        self._pointee_model = dmm.lookup(fe_type.dtype)
        self._pointee_be_type = self._pointee_model.get_data_type()
        be_type = self._pointee_be_type.as_pointer()
        super(PointerModel, self).__init__(dmm, fe_type, be_type)


@register_default(types.EphemeralPointer)
class EphemeralPointerModel(PointerModel):

    def get_data_type(self):
        return self._pointee_be_type

    def as_data(self, builder, value):
        value = builder.load(value)
        return self._pointee_model.as_data(builder, value)

    def from_data(self, builder, value):
        raise NotImplementedError("use load_from_data_pointer() instead")

    def load_from_data_pointer(self, builder, ptr):
        return builder.bitcast(ptr, self.get_value_type())


@register_default(types.EphemeralArray)
class EphemeralArrayModel(PointerModel):

    def get_data_type(self):
        return ir.ArrayType(self._pointee_be_type, self._fe_type.count)

    def as_data(self, builder, value):
        values = [builder.load(cgutils.gep(builder, value, i))
                  for i in range(self._fe_type.count)]
        return cgutils.pack_array(builder, values)

    def from_data(self, builder, value):
        raise NotImplementedError("use load_from_data_pointer() instead")

    def load_from_data_pointer(self, builder, ptr):
        return builder.bitcast(ptr, self.get_value_type())


@register_default(types.ExternalFunctionPointer)
class ExternalFuncPointerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        sig = fe_type.sig
        # Since the function is non-Numba, there is no adaptation
        # of arguments and return value, hence get_value_type().
        retty = dmm.lookup(sig.return_type).get_value_type()
        args = [dmm.lookup(t).get_value_type() for t in sig.args]
        be_type = ir.PointerType(ir.FunctionType(retty, args))
        super(ExternalFuncPointerModel, self).__init__(dmm, fe_type, be_type)


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
            v = builder.extract_value(value, [i])
            v = self._elem_model.as_argument(builder, v)
            out.append(v)
        return out

    def from_argument(self, builder, value):
        out = ir.Constant(self.get_value_type(), ir.Undefined)
        for i, v in enumerate(value):
            v = self._elem_model.from_argument(builder, v)
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

    def traverse(self, builder, value):
        values = cgutils.unpack_tuple(builder, value, count=self._count)
        return zip([self._fe_type.dtype] * len(values), values)


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

    def load_from_data_pointer(self, builder, ptr):
        values = []
        for i, model in enumerate(self._models):
            elem_ptr = cgutils.gep(builder, ptr, 0, i)
            val = model.load_from_data_pointer(builder, elem_ptr)
            values.append(val)

        struct = ir.Constant(self.get_value_type(), ir.Undefined)
        for i, val in enumerate(values):
            struct = self.set(builder, struct, val, i)
        return struct

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

    def traverse(self, builder, value):
        out = [(self.get_type(k), self.get(builder, value, k))
                for k in self._fields]
        return out


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
@register_default(types.Buffer)
@register_default(types.ByteArray)
@register_default(types.Bytes)
@register_default(types.MemoryView)
@register_default(types.PyArray)
class ArrayModel(StructModel):
    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            ('meminfo', types.meminfo_pointer),
            ('parent', types.pyobject),
            ('nitems', types.intp),
            ('itemsize', types.intp),
            ('data', types.CPointer(fe_type.dtype)),
            ('shape', types.UniTuple(types.intp, ndim)),
            ('strides', types.UniTuple(types.intp, ndim)),

        ]
        super(ArrayModel, self).__init__(dmm, fe_type, members)


@register_default(types.NestedArray)
class NestedArrayModel(ArrayModel):
    def __init__(self, dmm, fe_type):
        self._be_type = dmm.lookup(fe_type.dtype).get_data_type()
        super(NestedArrayModel, self).__init__(dmm, fe_type)

    def get_data_type(self):
        ret = ir.ArrayType(self._be_type, self._fe_type.nitems)
        return ret


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
        raise NotImplementedError

    def from_return(self, builder, value):
        return self._value_model.from_return(builder, value)

    def traverse(self, builder, value):
        data = self.get(builder, value, "data")
        valid = self.get(builder, value, "valid")
        data = builder.select(valid, data, ir.Constant(data.type, None))
        return [(self.get_type("data"), data),
                (self.get_type("valid"), valid)]


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

    def as_return(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value


class CContiguousFlatIter(StructModel):
    def __init__(self, dmm, fe_type, need_indices):
        assert fe_type.array_type.layout == 'C'
        array_type = fe_type.array_type
        dtype = array_type.dtype
        ndim = array_type.ndim
        members = [('array', types.EphemeralPointer(array_type)),
                   ('stride', types.intp),
                   ('pointer', types.EphemeralPointer(types.CPointer(dtype))),
                   ('index', types.EphemeralPointer(types.intp)),
                   ]
        if need_indices:
            # For ndenumerate()
            members.append(('indices', types.EphemeralArray(types.intp, ndim)))
        super(CContiguousFlatIter, self).__init__(dmm, fe_type, members)


class FlatIter(StructModel):
    def __init__(self, dmm, fe_type):
        array_type = fe_type.array_type
        dtype = array_type.dtype
        ndim = array_type.ndim
        members = [('array', types.EphemeralPointer(array_type)),
                   ('pointers', types.EphemeralArray(types.CPointer(dtype), ndim)),
                   ('indices', types.EphemeralArray(types.intp, ndim)),
                   ('exhausted', types.EphemeralPointer(types.boolean)),
        ]
        super(FlatIter, self).__init__(dmm, fe_type, members)


@register_default(types.UniTupleIter)
class UniTupleIter(StructModel):
    def __init__(self, dmm, fe_type):
        members = [('index', types.EphemeralPointer(types.intp)),
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
        members = [('index', types.EphemeralPointer(types.uintp)),
                   ('array', fe_type.array_type)]
        super(ArrayIterator, self).__init__(dmm, fe_type, members)


@register_default(types.EnumerateType)
class EnumerateType(StructModel):
    def __init__(self, dmm, fe_type):
        members = [('count', types.EphemeralPointer(types.intp)),
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
        members = [('iter', types.EphemeralPointer(int_type)),
                   ('stop', int_type),
                   ('step', int_type),
                   ('count', types.EphemeralPointer(int_type))]
        super(RangeIteratorType, self).__init__(dmm, fe_type, members)


@register_default(types.Generator)
class GeneratorModel(CompositeModel):
    def __init__(self, dmm, fe_type):
        super(GeneratorModel, self).__init__(dmm, fe_type)
        self._arg_models = [self._dmm.lookup(t) for t in fe_type.arg_types]
        self._state_models = [self._dmm.lookup(t) for t in fe_type.state_types]

        self._args_be_type = ir.LiteralStructType(
            [t.get_data_type() for t in self._arg_models])
        self._state_be_type = ir.LiteralStructType(
            [t.get_data_type() for t in self._state_models])
        # The whole generator closure
        self._be_type = ir.LiteralStructType(
            [self._dmm.lookup(types.int32).get_value_type(),
             self._args_be_type, self._state_be_type])

    def get_value_type(self):
        """
        The generator closure is passed around as a reference.
        """
        return self._be_type.as_pointer()

    def get_argument_type(self):
        return self.get_value_type()

    def get_return_type(self):
        return self._be_type

    def get_data_type(self):
        return self._be_type

    def as_argument(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value


@register_default(types.ArrayCTypes)
class ArrayCTypesModel(StructModel):
    def __init__(self, dmm, fe_type):
        # ndim = fe_type.ndim
        members = [('data', types.uintp)]
        super(ArrayCTypesModel, self).__init__(dmm, fe_type, members)


@register_default(types.RangeType)
class RangeModel(StructModel):
    def __init__(self, dmm, fe_type):
        int_type = fe_type.iterator_type.yield_type
        members = [('start', int_type),
                   ('stop', int_type),
                   ('step', int_type)]
        super(RangeModel, self).__init__(dmm, fe_type, members)


# =============================================================================

@register_default(types.NumpyNdIndexType)
class NdIndexType(StructModel):
    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [('shape', types.UniTuple(types.intp, ndim)),
                   ('indices', types.EphemeralArray(types.intp, ndim)),
                   ('exhausted', types.EphemeralPointer(types.boolean)),
                   ]
        super(NdIndexType, self).__init__(dmm, fe_type, members)


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

@register_default(types.BoundFunction)
def handle_bound_function(dmm, ty):
    # The same as the underlying type
    return dmm[ty.this]
