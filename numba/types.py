"""
These type objects do not have a fixed machine representation.  It is up to
the targets to choose their representation.
"""
from __future__ import print_function, division, absolute_import

import itertools
import struct
import weakref

import numpy

from .six import add_metaclass
from . import npdatetime, utils


# Types are added to a global registry (_typecache) in order to assign
# them unique integer codes for fast matching in _dispatcher.c.
# However, we also want types to be disposable, therefore we ensure
# each type is interned as a weak reference, so that it lives only as
# long as necessary to keep a stable type code.
_typecodes = itertools.count()

def _autoincr():
    n = next(_typecodes)
    # 4 billion types should be enough, right?
    assert n < 2 ** 32, "Limited to 4 billion types"
    return n

_typecache = {}

def _on_type_disposal(wr, _pop=_typecache.pop):
    _pop(wr, None)


class _TypeMetaclass(type):
    """
    A metaclass that will intern instances after they are created.
    This is done by first creating a new instance (including calling
    __init__, which sets up the required attributes for equality
    and hashing), then looking it up in the _typecache registry.
    """

    def __call__(cls, *args, **kwargs):
        """
        Instantiate *cls* (a Type subclass, presumably) and intern it.
        If an interned instance already exists, it is returned, otherwise
        the new instance is returned.
        """
        inst = type.__call__(cls, *args, **kwargs)
        # Try to intern the created instance
        wr = weakref.ref(inst, _on_type_disposal)
        orig = _typecache.get(wr)
        orig = orig and orig()
        if orig is not None:
            return orig
        else:
            inst._code = _autoincr()
            _typecache[wr] = wr
            inst.post_init()
            return inst


@add_metaclass(_TypeMetaclass)
class Type(object):
    """
    The default behavior is to provide equality through `name` attribute.
    Two types are equal if there `name` are equal.
    Subclass can refine this behavior.
    """

    mutable = False

    def __init__(self, name, param=False):
        self.name = name
        self.is_parametric = param

    def post_init(self):
        """
        A method called when the instance is fully initialized and has
        a registered typecode in its _code attribute.  Does nothing by
        default, but can be overriden.
        """

    @property
    def key(self):
        """
        A property used for __eq__, __ne__ and __hash__.  Can be overriden
        in subclasses.
        """
        return self.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.key == other.key

    def __ne__(self, other):
        return not (self == other)

    def __call__(self, *args):
        if len(args) == 1 and not isinstance(args[0], Type):
            return self.cast_python_value(args[0])
        return Prototype(args=args, return_type=self)

    def __getitem__(self, args):
        assert not isinstance(self, Array)
        ndim, layout = self._determine_array_spec(args)
        return Array(dtype=self, ndim=ndim, layout=layout)

    def _determine_array_spec(self, args):
        if isinstance(args, (tuple, list)):
            ndim = len(args)
            if args[0].step == 1:
                layout = 'F'
            elif args[-1].step == 1:
                layout = 'C'
            else:
                layout = 'A'
        elif isinstance(args, slice):
            ndim = 1
            if args.step == 1:
                layout = 'C'
            else:
                layout = 'A'
        else:
            ndim = 1
            layout = 'A'

        return ndim, layout


    __iter__ = NotImplemented
    cast_python_value = NotImplemented

    def coerce(self, typingctx, other):
        """Override this method to implement specialized coercion logic
        for extending unify_pairs().  Only use this if the coercion logic cannot
        be expressed as simple casting rules.
        """
        return NotImplemented


class OpaqueType(Type):
    """
    To deal with externally defined literal types
    """

    def __init__(self, name):
        super(OpaqueType, self).__init__(name)


class Boolean(Type):

    def cast_python_value(self, value):
        return bool(value)


class Number(Type):
    """
    Base class for number types.
    """


@utils.total_ordering
class Integer(Number):
    def __init__(self, *args, **kws):
        super(Integer, self).__init__(*args, **kws)
        # Determine bitwidth
        for prefix in ('int', 'uint'):
            if self.name.startswith(prefix):
                bitwidth = int(self.name[len(prefix):])
        self.bitwidth = bitwidth
        self.signed = self.name.startswith('int')

    def cast_python_value(self, value):
        return getattr(numpy, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        if self.signed != other.signed:
            return NotImplemented
        return self.bitwidth < other.bitwidth


@utils.total_ordering
class Float(Number):
    def __init__(self, *args, **kws):
        super(Float, self).__init__(*args, **kws)
        # Determine bitwidth
        assert self.name.startswith('float')
        bitwidth = int(self.name[5:])
        self.bitwidth = bitwidth

    def cast_python_value(self, value):
        return getattr(numpy, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth


@utils.total_ordering
class Complex(Number):
    def __init__(self, name, underlying_float, **kwargs):
        super(Complex, self).__init__(name, **kwargs)
        self.underlying_float = underlying_float
        # Determine bitwidth
        assert self.name.startswith('complex')
        bitwidth = int(self.name[7:])
        self.bitwidth = bitwidth

    def cast_python_value(self, value):
        return getattr(numpy, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth


class _NPDatetimeBase(Type):
    """
    Common base class for numpy.datetime64 and numpy.timedelta64.
    """

    def __init__(self, unit, *args, **kws):
        name = '%s(%s)' % (self.type_name, unit)
        self.unit = unit
        self.unit_code = npdatetime.DATETIME_UNITS[self.unit]
        super(_NPDatetimeBase, self).__init__(name, *args, **kws)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        # A coarser-grained unit is "smaller", i.e. less precise values
        # can be represented (but the magnitude of representable values is
        # also greater...).
        return self.unit_code < other.unit_code


@utils.total_ordering
class NPTimedelta(_NPDatetimeBase):
    type_name = 'timedelta64'

@utils.total_ordering
class NPDatetime(_NPDatetimeBase):
    type_name = 'datetime64'


class Prototype(Type):
    def __init__(self, args, return_type):
        self.args = args
        self.return_type = return_type
        name = "%s(%s)" % (return_type, ', '.join(str(a) for a in args))
        super(Prototype, self).__init__(name=name)


class Dummy(Type):
    """
    For type that does not really have a representation and is compatible
    with a void*.
    """
    pass


class Phantom(Dummy):
    """
    A type that cannot be materialized.  A Phantom cannot be used as
    argument or return type.
    """
    pass


class Opaque(Dummy):
    """
    A type that is a opaque pointer.
    """
    pass


class Kind(Type):
    def __init__(self, of):
        self.of = of
        super(Kind, self).__init__("kind(%s)" % of)

    @property
    def key(self):
        return self.of


class VarArg(Type):
    """
    Special type representing a variable number of arguments at the
    end of a function's signature.  Only used for signature matching,
    not for actual values.
    """

    def __init__(self, dtype):
        self.dtype = dtype
        super(VarArg, self).__init__("*%s" % dtype)

    @property
    def key(self):
        return self.dtype


class Module(Dummy):
    def __init__(self, pymod):
        self.pymod = pymod
        super(Module, self).__init__("Module(%s)" % pymod)

    @property
    def key(self):
        return self.pymod


class Macro(Type):
    def __init__(self, template):
        self.template = template
        cls = type(self)
        super(Macro, self).__init__("%s(%s)" % (cls.__name__, template))

    @property
    def key(self):
        return self.template


class Callable(Type):
    """
    Base class for callables.
    """

    def get_call_type(self, context, args, kws):
        """
        Using the typing *context*, resolve the callable's signature for
        the given arguments.  A signature object is returned, or None.
        """
        raise NotImplementedError


class Function(Callable, Opaque):
    def __init__(self, template):
        self.template = template
        name = "%s(%s)" % (self.__class__.__name__, template.key)
        super(Function, self).__init__(name)

    @property
    def key(self):
        return self.template

    def get_call_type(self, context, args, kws):
        return self.template(context).apply(args, kws)


class DTypeSpec(Opaque):
    """
    Base class for types usable as "dtype" arguments to various Numpy APIs
    (e.g. np.empty()).
    """

    @property
    def dtype(self):
        raise NotImplementedError


class NumberClass(Callable, DTypeSpec):
    """
    Type class for number classes (e.g. "np.float64").
    """

    def __init__(self, instance_type, template):
        self.instance_type = instance_type
        self.template = template
        name = "type(%s)" % (instance_type,)
        super(NumberClass, self).__init__(name)

    def get_call_type(self, context, args, kws):
        return self.template(context).apply(args, kws)

    @property
    def dtype(self):
        return self.instance_type


class DType(DTypeSpec):
    """
    Type class for Numpy dtypes.
    """

    def __init__(self, dtype):
        assert isinstance(dtype, Type)
        self._dtype = dtype
        name = "dtype(%s)" % (dtype,)
        super(DTypeSpec, self).__init__(name)

    @property
    def dtype(self):
        return self._dtype


class WeakType(Type):
    """
    Base class for types parametered by a mortal object, to which only
    a weak reference is kept.
    """

    def _store_object(self, obj):
        self._wr = weakref.ref(obj)

    def _get_object(self):
        obj = self._wr()
        if obj is None:
            raise ReferenceError("underlying object has vanished")
        return obj

    @property
    def key(self):
        return self._wr

    def __eq__(self, other):
        if type(self) is type(other):
            obj = self._wr()
            return obj is not None and obj is other._wr()

    def __hash__(self):
        return Type.__hash__(self)


class Dispatcher(WeakType, Callable, Dummy):

    def __init__(self, overloaded):
        self._store_object(overloaded)
        super(Dispatcher, self).__init__("Dispatcher(%s)" % overloaded)

    def get_call_type(self, context, args, kws):
        template, args, kws = self.overloaded.get_call_template(args, kws)
        sig = template(context).apply(args, kws)
        sig.pysig = self.pysig
        return sig

    @property
    def overloaded(self):
        """
        A strong reference to the underlying Dispatcher instance.
        """
        return self._get_object()

    @property
    def pysig(self):
        """
        A inspect.Signature object corresponding to this type.
        """
        return self.overloaded._pysig


class ExternalFunctionPointer(Function):
    """
    A pointer to a native function (e.g. exported via ctypes or cffi).
    *get_pointer* is a Python function taking an object
    and returning the raw pointer value as an int.
    """
    def __init__(self, sig, get_pointer, cconv=None):
        from .typing.templates import (AbstractTemplate, make_concrete_template,
                                       signature)
        if sig.return_type == ffi_forced_object:
            raise TypeError("Cannot return a pyobject from a external function")
        self.sig = sig
        self.requires_gil = any(a == ffi_forced_object for a in self.sig.args)
        self.get_pointer = get_pointer
        self.cconv = cconv
        if self.requires_gil:
            class GilRequiringDefn(AbstractTemplate):
                key = self.sig

                def generic(self, args, kws):
                    if kws:
                        raise TypeError("does not support keyword arguments")
                    # Make ffi_forced_object a bottom type to allow any type to be
                    # casted to it. This is the only place that support
                    # ffi_forced_object.
                    coerced = [actual if formal == ffi_forced_object else formal
                               for actual, formal
                               in zip(args, self.key.args)]
                    return signature(self.key.return_type, *coerced)
            template = GilRequiringDefn
        else:
            template = make_concrete_template("CFuncPtr", sig, [sig])
        super(ExternalFunctionPointer, self).__init__(template)

    @property
    def key(self):
        return self.sig, self.cconv, self.get_pointer


class ExternalFunction(Function):
    """
    A named native function (resolvable by LLVM).
    """

    def __init__(self, symbol, sig):
        from . import typing
        self.symbol = symbol
        self.sig = sig
        template = typing.make_concrete_template(symbol, symbol, [sig])
        super(ExternalFunction, self).__init__(template)


class BoundFunction(Function):
    def __init__(self, template, this):
        self.this = this
        newcls = type(template.__name__ + '.' + str(this), (template,),
                      dict(this=this))
        super(BoundFunction, self).__init__(newcls)

    @property
    def key(self):
        return (self.template.__name__, self.this)


class Method(Function):
    def __init__(self, template, this):
        self.this = this
        newcls = type(template.__name__ + '.' + str(this), (template,),
                      dict(this=this))
        super(Method, self).__init__(newcls)

    @property
    def key(self):
        return (self.template.__name__, self.this)


class Pair(Type):
    """
    A heterogenous pair.
    """

    def __init__(self, first_type, second_type):
        self.first_type = first_type
        self.second_type = second_type
        name = "pair<%s, %s>" % (first_type, second_type)
        super(Pair, self).__init__(name=name)

    @property
    def key(self):
        return self.first_type, self.second_type


class IterableType(Type):
    """
    Base class for iterable types.
    Derived classes should implement the *iterator_type* attribute.
    """


class SimpleIterableType(IterableType):

    def __init__(self, name, iterator_type):
        self.iterator_type = iterator_type
        super(SimpleIterableType, self).__init__(name, param=True)


class IteratorType(IterableType):
    """
    Base class for all iterator types.
    Derived classes should implement the *yield_type* attribute.
    """

    def __init__(self, name, **kwargs):
        self.iterator_type = self
        super(IteratorType, self).__init__(name, **kwargs)


class SimpleIteratorType(IteratorType):

    def __init__(self, name, yield_type):
        self.yield_type = yield_type
        super(SimpleIteratorType, self).__init__(name, param=True)


class RangeType(SimpleIterableType):
    pass


class RangeIteratorType(SimpleIteratorType):
    pass


class Generator(IteratorType):
    """
    Type class for Numba-compiled generator objects.
    """

    def __init__(self, gen_func, yield_type, arg_types, state_types,
                 has_finalizer):
        self.gen_func = gen_func
        self.arg_types = tuple(arg_types)
        self.state_types = tuple(state_types)
        self.yield_type = yield_type
        self.has_finalizer = has_finalizer
        name = "%s generator(func=%s, args=%s, has_finalizer=%s)" % (
            self.yield_type, self.gen_func, self.arg_types,
            self.has_finalizer)
        super(Generator, self).__init__(name, param=True)

    @property
    def key(self):
        return self.gen_func, self.arg_types, self.yield_type, self.has_finalizer


class NumpyFlatType(IteratorType):
    """
    Type class for `ndarray.flat()` objects.
    """

    def __init__(self, arrty):
        self.array_type = arrty
        self.yield_type = arrty.dtype
        name = "array.flat({arrayty})".format(arrayty=arrty)
        super(NumpyFlatType, self).__init__(name, param=True)

    @property
    def key(self):
        return self.array_type


class NumpyNdEnumerateType(IteratorType):
    """
    Type class for `np.ndenumerate()` objects.
    """

    def __init__(self, arrty):
        self.array_type = arrty
        # XXX making this a uintp has the side effect of forcing some
        # arithmetic operations to return a float result.
        self.yield_type = Tuple((UniTuple(intp, arrty.ndim), arrty.dtype))
        name = "ndenumerate({arrayty})".format(arrayty=arrty)
        super(NumpyNdEnumerateType, self).__init__(name, param=True)

    @property
    def key(self):
        return self.array_type


class NumpyNdIndexType(IteratorType):
    """
    Type class for `np.ndindex()` objects.
    """

    def __init__(self, ndim):
        self.ndim = ndim
        self.yield_type = UniTuple(intp, self.ndim)
        name = "ndindex(dims={ndim})".format(ndim=ndim)
        super(NumpyNdIndexType, self).__init__(name, param=True)

    @property
    def key(self):
        return self.ndim


class EnumerateType(IteratorType):
    """
    Type class for `enumerate` objects.
    Type instances are parametered with the underlying source type.
    """

    def __init__(self, iterable_type):
        self.source_type = iterable_type.iterator_type
        self.yield_type = Tuple([intp, self.source_type.yield_type])
        name = 'enumerate(%s)' % (self.source_type)
        super(EnumerateType, self).__init__(name, param=True)

    @property
    def key(self):
        return self.source_type


class ZipType(IteratorType):
    """
    Type class for `zip` objects.
    Type instances are parametered with the underlying source types.
    """

    def __init__(self, iterable_types):
        self.source_types = tuple(tp.iterator_type for tp in iterable_types)
        self.yield_type = Tuple(tp.yield_type for tp in self.source_types)
        name = 'zip(%s)' % ', '.join(str(tp) for tp in self.source_types)
        super(ZipType, self).__init__(name, param=True)

    @property
    def key(self):
        return self.source_types


class CharSeq(Type):
    mutable = True

    def __init__(self, count):
        self.count = count
        name = "[char x %d]" % count
        super(CharSeq, self).__init__(name, param=True)

    @property
    def key(self):
        return self.count


class UnicodeCharSeq(Type):
    mutable = True

    def __init__(self, count):
        self.count = count
        name = "[unichr x %d]" % count
        super(UnicodeCharSeq, self).__init__(name, param=True)

    @property
    def key(self):
        return self.count


class Record(Type):
    mutable = True

    def __init__(self, id, fields, size, aligned, dtype):
        self.id = id
        self.fields = fields.copy()
        self.size = size
        self.aligned = aligned
        self.dtype = dtype
        name = 'Record(%s)' % id
        super(Record, self).__init__(name)

    @property
    def key(self):
        return (self.dtype, self.size, self.aligned)

    def __len__(self):
        return len(self.fields)

    def offset(self, key):
        return self.fields[key][1]

    def typeof(self, key):
        return self.fields[key][0]

    @property
    def members(self):
        return [(f, t) for f, (t, _) in self.fields.items()]


class ArrayIterator(IteratorType):
    """
    Type class for iterators of array and buffer objects.
    """

    def __init__(self, array_type):
        self.array_type = array_type
        name = "iter(%s)" % (self.array_type,)
        nd = array_type.ndim
        if nd == 0 or nd == 1:
            self.yield_type = array_type.dtype
        else:
            self.yield_type = array_type.copy(ndim=array_type.ndim - 1)
        super(ArrayIterator, self).__init__(name, param=True)


class Buffer(IterableType):
    """
    Type class for objects providing the buffer protocol.
    Derived classes exist for more specific cases.
    """
    mutable = True
    slice_is_copy = False

    # CS and FS are not reserved for inner contig but strided
    LAYOUTS = frozenset(['C', 'F', 'CS', 'FS', 'A'])

    def __init__(self, dtype, ndim, layout, readonly=False, name=None):
        if isinstance(dtype, Buffer):
            raise TypeError("Buffer dtype cannot be buffer")
        if layout not in self.LAYOUTS:
            raise ValueError("Invalid layout '%s'" % layout)
        self.dtype = dtype
        self.ndim = ndim
        self.layout = layout
        if readonly:
            self.mutable = False
        if name is None:
            type_name = self.__class__.__name__.lower()
            if readonly:
                type_name = "readonly %s" % type_name
            name = "%s(%s, %sd, %s)" % (type_name, dtype, ndim, layout)
        super(Buffer, self).__init__(name, param=True)
        self.iterator_type = ArrayIterator(self)

    def copy(self, dtype=None, ndim=None, layout=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        return self.__class__(dtype=dtype, ndim=ndim, layout=layout,
                              readonly=not self.mutable)

    @property
    def key(self):
        return self.dtype, self.ndim, self.layout, self.mutable

    @property
    def is_c_contig(self):
        return self.layout == 'C' or (self.ndim <= 1 and self.layout in 'CF')

    @property
    def is_f_contig(self):
        return self.layout == 'F' or (self.ndim <= 1 and self.layout in 'CF')

    @property
    def is_contig(self):
        return self.layout in 'CF'


class Bytes(Buffer):
    """
    Type class for Python 3.x bytes objects.
    """
    mutable = False
    # Actually true but doesn't matter since bytes is immutable
    slice_is_copy = False


class ByteArray(Buffer):
    """
    Type class for bytearray objects.
    """
    slice_is_copy = True


class PyArray(Buffer):
    """
    Type class for array.array objects.
    """
    slice_is_copy = True


class MemoryView(Buffer):
    """
    Type class for memoryview objects.
    """


class Array(Buffer):
    """
    Type class for Numpy arrays.
    """

    def __init__(self, dtype, ndim, layout, readonly=False, name=None):
        if readonly:
            self.mutable = False
        if name is None:
            type_name = "array" if self.mutable else "readonly array"
            name = "%s(%s, %sd, %s)" % (type_name, dtype, ndim, layout)
        super(Array, self).__init__(dtype, ndim, layout, name=name)

    def post_init(self):
        """
        Install conversion from this layout (if non-'A') to 'A' layout.
        """
        if self.layout != 'A':
            from numba.typeconv.rules import default_casting_rules as tcr
            ary_any = self.copy(layout='A')
            # XXX This will make the types immortal
            tcr.safe(self, ary_any)

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        return Array(dtype=dtype, ndim=ndim, layout=layout, readonly=readonly)

    @property
    def key(self):
        return self.dtype, self.ndim, self.layout, self.mutable


class ArrayCTypes(Type):
    """
    This is the type for `numpy.ndarray.ctypes`.
    """
    def __init__(self, arytype):
        # This depends on the ndim for the shape and strides attributes,
        # even though they are not implemented, yet.
        self.ndim = arytype.ndim
        name = "ArrayCType(ndim={0})".format(self.ndim)
        super(ArrayCTypes, self).__init__(name, param=True)

    @property
    def key(self):
        return self.ndim


class NestedArray(Array):
    """
    A NestedArray is an array nested within a structured type (which are "void"
    type in NumPy parlance). Unlike an Array, the shape, and not just the number
    of dimenions is part of the type of a NestedArray.
    """

    def __init__(self, dtype, shape):
        assert dtype.bitwidth % 8 == 0, \
            "Dtype bitwidth must be a multiple of bytes"
        self._shape = shape
        name = "nestedarray(%s, %s)" % (dtype, shape)
        ndim = len(shape)
        super(NestedArray, self).__init__(dtype, ndim, 'C', name=name)

    @property
    def shape(self):
        return self._shape

    @property
    def nitems(self):
        l = 1
        for s in self.shape:
            l = l * s
        return l

    @property
    def size(self):
        return self.dtype.bitwidth // 8

    @property
    def strides(self):
        stride = self.size
        strides = []
        for i in reversed(self._shape):
             strides.append(stride)
             stride *= i
        return tuple(reversed(strides))

    @property
    def key(self):
        return self.dtype, self.shape


class BaseTuple(Type):
    """
    The base class for all tuple types (with a known size).
    """


class UniTuple(IterableType, BaseTuple):

    def __init__(self, dtype, count):
        self.dtype = dtype
        self.count = count
        name = "(%s x %d)" % (dtype, count)
        super(UniTuple, self).__init__(name, param=True)
        self.iterator_type = UniTupleIter(self)

    def getitem(self, ind):
        return self.dtype, intp

    def __getitem__(self, i):
        """
        Return element at position i
        """
        return self.dtype

    def __iter__(self):
        return iter([self.dtype] * self.count)

    def __len__(self):
        return self.count

    @property
    def key(self):
        return self.dtype, self.count

    @property
    def types(self):
        return (self.dtype,) * self.count

    def coerce(self, typingctx, other):
        """
        Unify UniTuples with their dtype
        """
        if isinstance(other, UniTuple) and len(self) == len(other):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            return UniTuple(dtype=dtype, count=self.count)

        return NotImplemented


class UniTupleIter(IteratorType):

    def __init__(self, unituple):
        self.unituple = unituple
        self.yield_type = unituple.dtype
        name = 'iter(%s)' % unituple
        super(UniTupleIter, self).__init__(name, param=True)

    @property
    def key(self):
        return self.unituple


class Tuple(BaseTuple):

    def __init__(self, types):
        self.types = tuple(types)
        self.count = len(self.types)
        name = "(%s)" % ', '.join(str(i) for i in self.types)
        super(Tuple, self).__init__(name, param=True)

    def __getitem__(self, i):
        """
        Return element at position i
        """
        return self.types[i]

    def __len__(self):
        # Beware: this makes Tuple(()) false-ish
        return len(self.types)

    @property
    def key(self):
        return self.types

    def __iter__(self):
        return iter(self.types)

    def coerce(self, typingctx, other):
        """
        Unify elements of Tuples/UniTuples
        """
        # Other is UniTuple or Tuple
        if isinstance(other, (UniTuple, Tuple)) and len(self) == len(other):
            unified = [typingctx.unify_pairs(ta, tb)
                       for ta, tb in zip(self, other)]

            if any(t == pyobject for t in unified):
                return NotImplemented

            return Tuple(unified)

        return NotImplemented


class CPointer(Type):
    """
    Type class for pointers to other types.
    """
    mutable = True

    def __init__(self, dtype):
        self.dtype = dtype
        name = "*%s" % dtype
        super(CPointer, self).__init__(name, param=True)

    @property
    def key(self):
        return self.dtype


class EphemeralPointer(CPointer):
    """
    Type class for pointers which aren't guaranteed to last long - e.g.
    stack-allocated slots.  The data model serializes such pointers
    by copying the data pointed to.
    """


class EphemeralArray(Type):
    """
    Similar to EphemeralPointer, but pointing to an array of elements,
    rather than a single one.  The array size must be known at compile-time.
    """

    def __init__(self, dtype, count):
        self.dtype = dtype
        self.count = count
        name = "*%s[%d]" % (dtype, count)
        super(EphemeralArray, self).__init__(name, param=True)

    @property
    def key(self):
        return self.dtype, self.count


class Object(Type):
    mutable = True

    def __init__(self, clsobj):
        self.cls = clsobj
        name = "Object(%s)" % clsobj.__name__
        super(Object, self).__init__(name, param=True)

    @property
    def key(self):
        return self.cls


class Optional(Type):
    def __init__(self, typ):
        assert typ != none
        assert not isinstance(typ, Optional)
        self.type = typ
        name = "?%s" % typ
        super(Optional, self).__init__(name, param=True)

    def post_init(self):
        """
        Install conversion from optional(T) to T
        """
        from numba.typeconv.rules import default_casting_rules as tcr
        tcr.safe(self, self.type)
        tcr.promote(self.type, self)
        tcr.promote(none, self)

    @property
    def key(self):
        return self.type

    def coerce(self, typingctx, other):
        if isinstance(other, Optional):
            unified = typingctx.unify_pairs(self.type, other.type)

        else:
            unified = typingctx.unify_pairs(self.type, other)

        if unified != pyobject:
            return Optional(unified)

        return NotImplemented


class NoneType(Opaque):
    def coerce(self, typingctx, other):
        """Turns anything to a Optional type
        """
        if isinstance(other, Optional):
            return other

        return Optional(other)


class ExceptionType(Callable, Phantom):
    """
    The type of exception classes (not instances).
    """

    def __init__(self, exc_class):
        assert issubclass(exc_class, BaseException)
        name = "%s" % (exc_class.__name__)
        self.exc_class = exc_class
        super(ExceptionType, self).__init__(name, param=True)

    def get_call_type(self, context, args, kws):
        from . import typing
        return_type = ExceptionInstance(self.exc_class)
        return typing.signature(return_type)

    @property
    def key(self):
        return self.exc_class


class ExceptionInstance(Phantom):
    """
    The type of exception instances.  *exc_class* should be the
    exception class.
    """

    def __init__(self, exc_class):
        assert issubclass(exc_class, BaseException)
        name = "%s(...)" % (exc_class.__name__,)
        self.exc_class = exc_class
        super(ExceptionInstance, self).__init__(name, param=True)

    @property
    def key(self):
        return self.exc_class


class Slice3Type(Type):
    pass




# Utils

def is_int_tuple(x):
    if isinstance(x, Tuple):
        return all(i in integer_domain for i in x.types)
    elif isinstance(x, UniTuple):
        return x.dtype in integer_domain
    else:
        return False


# Short names


pyobject = Opaque('pyobject')
ffi_forced_object = Opaque('ffi_forced_object')
none = NoneType('none')
Any = Phantom('any')
string = Dummy('str')

# No operation is defined on voidptr
# Can only pass it around
voidptr = Opaque('void*')

# For NRT GC
meminfo_pointer = Opaque("MemInfo*")

boolean = bool_ = Boolean('bool')

byte = uint8 = Integer('uint8')
uint16 = Integer('uint16')
uint32 = Integer('uint32')
uint64 = Integer('uint64')

int8 = Integer('int8')
int16 = Integer('int16')
int32 = Integer('int32')
int64 = Integer('int64')
intp = int32 if utils.MACHINE_BITS == 32 else int64
uintp = uint32 if utils.MACHINE_BITS == 32 else uint64
intc = int32 if struct.calcsize('i') == 4 else int64
uintc = uint32 if struct.calcsize('i') == 4 else uint64

float32 = Float('float32')
float64 = Float('float64')

complex64 = Complex('complex64', float32)
complex128 = Complex('complex128', float64)

len_type = Phantom('len')
range_type = Phantom('range')
slice_type = Phantom('slice')
abs_type = Phantom('abs')
neg_type = Phantom('neg')
print_type = Phantom('print')
print_item_type = Phantom('print-item')
sign_type = Phantom('sign')

range_iter32_type = RangeIteratorType('range_iter32', int32)
range_iter64_type = RangeIteratorType('range_iter64', int64)
unsigned_range_iter64_type = RangeIteratorType('unsigned_range_iter64', uint64)
range_state32_type = RangeType('range_state32', range_iter32_type)
range_state64_type = RangeType('range_state64', range_iter64_type)
unsigned_range_state64_type = RangeType('unsigned_range_state64',
                                        unsigned_range_iter64_type)

# slice2_type = Type('slice2_type')
slice3_type = Slice3Type('slice3_type')

signed_domain = frozenset([int8, int16, int32, int64])
unsigned_domain = frozenset([uint8, uint16, uint32, uint64])
integer_domain = signed_domain | unsigned_domain
real_domain = frozenset([float32, float64])
complex_domain = frozenset([complex64, complex128])
number_domain = real_domain | integer_domain | complex_domain

# Aliases to Numpy type names

b1 = bool_
i1 = int8
i2 = int16
i4 = int32
i8 = int64
u1 = uint8
u2 = uint16
u4 = uint32
u8 = uint64

f4 = float32
f8 = float64

c8 = complex64
c16 = complex128

float_ = float32
double = float64
void = none

_make_signed = lambda x: globals()["int%d" % (numpy.dtype(x).itemsize * 8)]
_make_unsigned = lambda x: globals()["uint%d" % (numpy.dtype(x).itemsize * 8)]

char = _make_signed(numpy.byte)
uchar = byte = _make_unsigned(numpy.byte)
short = _make_signed(numpy.short)
ushort = _make_unsigned(numpy.short)
int_ = _make_signed(numpy.int_)
uint = _make_unsigned(numpy.int_)
intc = _make_signed(numpy.intc) # C-compat int
uintc = _make_unsigned(numpy.uintc) # C-compat uint
long_ = _make_signed(numpy.long)
ulong = _make_unsigned(numpy.long)
longlong = _make_signed(numpy.longlong)
ulonglong = _make_unsigned(numpy.longlong)

# optional types
optional = Optional


def is_numeric(ty):
    return ty in number_domain

_type_promote_map = {
    int8: (int16, False),
    uint8: (uint16, False),
    int16: (int32, False),
    uint16: (uint32, False),
    int32: (int64, False),
    uint32: (uint64, False),
    int64: (float64, True),
    uint64: (float64, True),
    float32: (float64, False),
    complex64: (complex128, True),
}


def promote_numeric_type(ty):
    res = _type_promote_map.get(ty)
    if res is None:
        if ty not in number_domain:
            raise TypeError(ty)
        else:
            return None, None  # no promote available

    return res


__all__ = '''
int8
int16
int32
int64
uint8
uint16
uint32
uint64
intp
uintp
intc
uintc
boolean
float32
float64
complex64
complex128
bool_
byte
char
uchar
short
ushort
int_
uint
long_
ulong
longlong
ulonglong
float_
double
void
none
b1
i1
i2
i4
i8
u1
u2
u4
u8
f4
f8
c8
c16
optional
ffi_forced_object
'''.split()
