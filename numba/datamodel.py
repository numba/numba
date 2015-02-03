from __future__ import print_function, absolute_import

import functools
from collections import deque
from llvmlite import ir
from numba import types


class DataModelManager(object):
    """Manages mapping of FE types to their corresponding data model
    """

    def __init__(self):
        # handler map
        # key: numba.types.Type subclass
        # value: function
        self._handlers = {}

    def register(self, fetypecls, handler):
        assert issubclass(fetypecls, types.Type)
        self._handlers[fetypecls] = handler

    def lookup(self, fetype):
        handler = self._handlers[type(fetype)]
        return handler(self, fetype)


class FunctionInfo(object):
    def __init__(self, dmm, fe_ret, fe_args):
        self._dmm = dmm
        self._fe_ret = fe_ret
        self._fe_args = fe_args
        self._nargs = len(fe_args)
        self._dm_ret = self._dmm.lookup(fe_ret)
        self._dm_args = [self._dmm.lookup(ty) for ty in fe_args]
        argtys = [bt.get_argument_type() for bt in self._dm_args]
        self._be_args, self._posmap = zip(*_flatten(argtys))

    def as_arguments(self, builder, values):
        if len(values) != self._nargs:
            raise TypeError("invalid number of args")

        args = [dm.as_argument(builder, val)
                for dm, val in zip(self._dm_args, values)]

        args, _ = zip(*_flatten(args))
        return args

    def reverse_as_arguments(self, builder, args):
        if len(args) != len(self._posmap):
            raise TypeError("invalid number of args")

        valtree = _unflatten(self._posmap, args)

        values = [dm.reverse_as_argument(builder, val)
                  for dm, val in zip(self._dm_args, valtree)]

        return values

    @property
    def argument_types(self):
        return tuple(self._be_args)


def _unflatten(posmap, flatiter):
    if 0 == len(posmap):
        raise StopIteration

    poss = deque(posmap)
    vals = deque(flatiter)

    assert len(vals) == len(poss)

    depth = len(poss[0])
    last = None
    while poss:
        cur = poss[0]
        # Depth increased
        if len(cur) > depth:
            ret = tuple(_unflatten(poss, vals))
            yield ret
            # skip processed data in the recursive call
            for _ in range(len(ret)):
                vals.popleft()
                poss.popleft()
        # Depth decreased
        elif len(cur) < depth:
            raise StopIteration
        # Depth unchanged but new sequence
        elif last is not None and last[:-1] != cur[:-1]:
            raise StopIteration
        # Depth unchanged and continue sequence
        else:
            yield vals.popleft()
            last = poss.popleft()


def _flatten(iterable, indices=(0,)):
    """
    Flatten nested iterable of (tuple, list) with position information
    """
    for i in iterable:
        if isinstance(i, (tuple, list)):
            inner = indices + (0,)
            for j, k in _flatten(i, indices=inner):
                yield j, k
        else:
            yield i, indices
        indices = indices[:-1] + (indices[-1] + 1,)


def register(dmm, typecls):
    def wraps(fn):
        dmm.register(typecls, fn)
        return fn

    return wraps


defaultDataModelManager = DataModelManager()

register_default = functools.partial(register, defaultDataModelManager)


@register_default(types.Integer)
def handle_integers(dmm, ty):
    return IntegerModel(dmm, ty)


@register_default(types.Float)
def handle_floats(dmm, ty):
    if ty == types.float32:
        return FloatModel(dmm)
    elif ty == types.float64:
        return DoubleModel(dmm)
    else:
        raise NotImplementedError(ty)


@register_default(types.Complex)
def handle_complex_numbers(dmm, ty):
    if ty == types.complex64:
        return ComplexModel(dmm)
    elif ty == types.complex128:
        return DoubleComplexModel(dmm)
    else:
        raise NotImplementedError(ty)


@register_default(types.CPointer)
def handle_pointer(dmm, ty):
    return PointerModel(dmm, ty)


@register_default(types.UniTuple)
def handle_unituple(dmm, ty):
    return UniTupleModel(dmm, ty)


@register_default(types.Array)
def handle_array(dmm, ty):
    return ArrayModel(dmm, ty)


# ============== Define Data Models ==============

class DataModel(object):
    def __init__(self, dmm):
        self._dmm = dmm

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

    def reverse_as_data(self, builder, value):
        return NotImplemented

    def reverse_as_argument(self, builder, value):
        return NotImplemented

    def reverse_as_return(self, builder, value):
        return NotImplemented

    def _compared_fields(self):
        """
        The default comparison uses the type(self).
        So any instance of the same type is equal.
        """
        return (type(self),)

    def __hash__(self):
        return hash(tuple(self._compared_fields()))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._compared_fields() == other._compared_fields()
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # ===== Helpers =====

    def data_to_argument(self, builder, value):
        return self.as_argument(builder, self.reverse_as_data(builder, value))

    def argument_to_data(self, builder, value):
        return self.as_data(builder, self.reverse_as_argument(builder, value))


class PrimitiveModel(DataModel):
    """A primitive type can be represented natively in the target in all
    usage contexts.
    """

    def __init__(self, dmm, be_type):
        super(PrimitiveModel, self).__init__(dmm)
        self.be_type = be_type

    def get_value_type(self):
        return self.be_type

    def as_data(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return self.as_data(builder, value)

    def as_return(self, builder, value):
        return self.as_data(builder, value)

    def reverse_as_data(self, builder, value):
        return value

    def reverse_as_argument(self, builder, value):
        return self.reverse_as_data(builder, value)

    def reverse_as_return(self, builder, value):
        return self.reverse_as_data(builder, value)

    def _compared_fields(self):
        return (self.be_type,)


class IntegerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        self.fe_type = fe_type
        self.signed = fe_type.signed
        be_type = ir.IntType(self.fe_type.bitwidth)
        super(IntegerModel, self).__init__(dmm, be_type)

    def _compared_fields(self):
        return self.be_type, self.signed


class FloatModel(PrimitiveModel):
    def __init__(self, dmm):
        super(FloatModel, self).__init__(dmm, ir.FloatType())


class DoubleModel(PrimitiveModel):
    def __init__(self, dmm):
        super(DoubleModel, self).__init__(dmm, ir.DoubleType())


class BaseComplexModel(DataModel):
    _element_type = NotImplemented

    def __init__(self, dmm):
        super(BaseComplexModel, self).__init__(dmm)
        self._element_model = self._dmm.lookup(self._element_type)

    def get_value_type(self):
        elem = self._element_model.get_data_type()
        return ir.LiteralStructType([elem] * 2)

    def get_argument_type(self):
        elem = self._element_model.get_data_type()
        return tuple([elem] * 2)

    def as_argument(self, builder, value):
        real = builder.extract_value(value, [0])
        imag = builder.extract_value(value, [1])

        real = self._element_model.data_to_argument(builder, real)
        imag = self._element_model.data_to_argument(builder, imag)

        return real, imag

    def reverse_as_argument(self, builder, value):
        real, imag = value
        real = self._element_model.argument_to_data(builder, real)
        imag = self._element_model.argument_to_data(builder, imag)
        valty = self.get_value_type()
        val = ir.Constant(valty, ir.Undefined)
        val = builder.insert_value(val, real, [0])
        val = builder.insert_value(val, imag, [1])
        return val

    def as_return(self, builder, value):
        return value

    def reverse_as_return(self, builder, value):
        return value

    def as_data(self, builder, value):
        return value

    def reverse_as_data(self, builder, value):
        return value


class ComplexModel(BaseComplexModel):
    _element_type = types.float32


class DoubleComplexModel(BaseComplexModel):
    _element_type = types.float64


class PointerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = dmm.lookup(fe_type.dtype).get_data_type().as_pointer()
        super(PointerModel, self).__init__(dmm, be_type)


class UniTupleModel(DataModel):
    def __init__(self, dmm, fe_type):
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

    def reverse_as_argument(self, builder, value):
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

    def reverse_as_data(self, builder, value):
        out = ir.Constant(self.get_value_type(), ir.Undefined)
        for i in range(self._count):
            val = builder.extract_value(value, [i])
            dval = self._elem_model.reverse_as_data(builder, val)
            out = builder.insert_value(out, dval, [i])
        return out

    def as_return(self, builder, value):
        return value

    def reverse_as_return(self, builder, value):
        return value


class ArrayModel(DataModel):
    def __init__(self, dmm, fe_type):
        super(ArrayModel, self).__init__(dmm)
        self.fe_type = fe_type
        self._ndim = self.fe_type.ndim
        self._dataptr_model = self._dmm.lookup(types.CPointer(fe_type.dtype))
        self._shape_model = self._dmm.lookup(types.UniTuple(types.intp,
                                                            self._ndim))
        self._strides_model = self._shape_model

    def get_value_type(self):
        elems = [
            self._dataptr_model.get_data_type(),
            self._shape_model.get_data_type(),
            self._strides_model.get_data_type(),
        ]
        return ir.LiteralStructType(elems)

    def get_argument_type(self):
        return (self._dataptr_model.get_argument_type(),
                self._shape_model.get_argument_type(),
                self._strides_model.get_argument_type(),)

    def get_return_type(self):
        return self.get_value_type()

    def as_argument(self, builder, value):
        data = builder.extract_value(value, [0])
        shapes = builder.extract_value(value, [1])
        strides = builder.extract_value(value, [2])
        data = self._dataptr_model.data_to_argument(builder, data)
        shapes = self._shape_model.data_to_argument(builder, shapes)
        strides = self._shape_model.data_to_argument(builder, strides)
        return data, shapes, strides

    def reverse_as_argument(self, builder, value):
        data, shapes, strides = value

        data = self._dataptr_model.argument_to_data(builder, data)
        shapes = self._shape_model.argument_to_data(builder, shapes)
        strides = self._shape_model.argument_to_data(builder, strides)

        val = ir.Constant(self.get_value_type(), ir.Undefined)
        val = builder.insert_value(val, data, [0])
        val = builder.insert_value(val, shapes, [1])
        val = builder.insert_value(val, strides, [2])
        return val

    def as_return(self, builder, value):
        return value

    def reverse_as_return(self, builder, value):
        return value

