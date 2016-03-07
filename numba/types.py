"""
These type objects do not have a fixed machine representation.  It is up to
the targets to choose their representation.
"""
from __future__ import print_function, division, absolute_import

import struct
import weakref

import numpy

# All abstract types are exposed through this module
from .abstracttypes import *
from . import npdatetime, utils
from .typeconv import Conversion


class Boolean(Hashable):

    def cast_python_value(self, value):
        return bool(value)


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

    @classmethod
    def from_bitwidth(cls, bitwidth, signed=True):
        name = ('int%d' if signed else 'uint%d') % bitwidth
        return globals()[name]

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

    def cast_python_value(self, value):
        cls = getattr(numpy, self.type_name)
        if self.unit:
            return cls(value, self.unit)
        else:
            return cls(value)


@utils.total_ordering
class NPTimedelta(_NPDatetimeBase):
    type_name = 'timedelta64'

@utils.total_ordering
class NPDatetime(_NPDatetimeBase):
    type_name = 'datetime64'


class Phantom(Dummy):
    """
    A type that cannot be materialized.  A Phantom cannot be used as
    argument or return type.
    """


class Undefined(Dummy):
    """
    A type that is left imprecise.  This is used as a temporaray placeholder
    during type inference in the hope that the type can be later refined.
    """

    def is_precise(self):
        return False


class Opaque(Dummy):
    """
    A type that is a opaque pointer.
    """


class PyObject(Dummy):
    """
    A generic CPython object.
    """

    def is_precise(self):
        return False


class RawPointer(Dummy):
    """
    A raw pointer without any specific meaning.
    """


class Const(Dummy):
    """
    A compile-time constant, for (internal) use when a type is needed for
    lookup.
    """

    def __init__(self, value):
        self.value = value
        # We want to support constants of non-hashable values, therefore
        # fall back on the value's id() if necessary.
        try:
            hash(value)
        except TypeError:
            self._key = id(value)
        else:
            self._key = value
        super(Const, self).__init__("const(%r)" % (value,))

    @property
    def key(self):
        return type(self.value), self._key


class Omitted(Opaque):
    """
    An omitted function argument with a default value.
    """

    def __init__(self, value):
        self.value = value
        super(Omitted, self).__init__("omitted(default=%r)" % (value,))

    @property
    def key(self):
        return type(self.value), id(self.value)


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


class Function(Callable, Opaque):
    """
    Base type class for some function types.
    """

    def __init__(self, template):
        if isinstance(template, (list, tuple)):
            self.templates = tuple(template)
            keys = set(temp.key for temp in self.templates)
            if len(keys) != 1:
                raise ValueError("incompatible templates: keys = %s"
                                 % (this,))
            self.typing_key, = keys
        else:
            self.templates = (template,)
            self.typing_key = template.key
        self._impl_keys = {}
        name = "%s(%s)" % (self.__class__.__name__, self.typing_key)
        super(Function, self).__init__(name)

    @property
    def key(self):
        return self.typing_key, self.templates

    def augment(self, other):
        """
        Augment this function type with the other function types' templates,
        so as to support more input types.
        """
        if type(other) is type(self) and other.typing_key == self.typing_key:
            return type(self)(self.templates + other.templates)

    def get_impl_key(self, sig):
        """
        Get the implementation key (used by the target context) for the
        given signature.
        """
        return self._impl_keys[sig.args]

    def get_call_type(self, context, args, kws):
        for temp_cls in self.templates:
            temp = temp_cls(context)
            sig = temp.apply(args, kws)
            if sig is not None:
                self._impl_keys[sig.args] = temp.get_impl_key(sig)
                return sig

    def get_call_signatures(self):
        sigs = []
        is_param = False
        for temp in self.templates:
            sigs += getattr(temp, 'cases', [])
            is_param = is_param or hasattr(temp, 'generic')
        return sigs, is_param


class BoundFunction(Callable, Opaque):
    """
    A function with an implicit first argument (denoted as *this* below).
    """

    def __init__(self, template, this):
        # Create a derived template with an attribute *this*
        newcls = type(template.__name__ + '.' + str(this), (template,),
                      dict(this=this))
        self.template = newcls
        self.typing_key = self.template.key
        self.this = this
        name = "%s(%s for %s)" % (self.__class__.__name__,
                                  self.typing_key, self.this)
        super(BoundFunction, self).__init__(name)

    def unify(self, typingctx, other):
        if (isinstance(other, BoundFunction) and
            self.typing_key == other.typing_key):
            this = typingctx.unify_pairs(self.this, other.this)
            if this != pyobject:
                # XXX is it right that both template instances are distinct?
                return self.copy(this=this)

    def copy(self, this):
        return type(self)(self.template, this)

    @property
    def key(self):
        return self.typing_key, self.this

    def get_impl_key(self, sig):
        """
        Get the implementation key (used by the target context) for the
        given signature.
        """
        return self.typing_key

    def get_call_type(self, context, args, kws):
        return self.template(context).apply(args, kws)

    def get_call_signatures(self):
        sigs = getattr(self.template, 'cases', [])
        is_param = hasattr(self.template, 'generic')
        return sigs, is_param


class NamedTupleClass(Callable, Opaque):
    """
    Type class for namedtuple classes.
    """

    def __init__(self, instance_class):
        self.instance_class = instance_class
        name = "class(%s)" % (instance_class)
        super(NamedTupleClass, self).__init__(name)

    def get_call_type(self, context, args, kws):
        # Overriden by the __call__ constructor resolution in typing.collections
        return None

    def get_call_signatures(self):
        return (), True

    @property
    def key(self):
        return self.instance_class


class NumberClass(Callable, DTypeSpec, Opaque):
    """
    Type class for number classes (e.g. "np.float64").
    """

    def __init__(self, instance_type):
        self.instance_type = instance_type
        name = "class(%s)" % (instance_type,)
        super(NumberClass, self).__init__(name)

    def get_call_type(self, context, args, kws):
        # Overriden by the __call__ constructor resolution in typing.builtins
        return None

    def get_call_signatures(self):
        return (), True

    @property
    def key(self):
        return self.instance_type

    @property
    def dtype(self):
        return self.instance_type


class DType(DTypeSpec, Opaque):
    """
    Type class for Numpy dtypes.
    """

    def __init__(self, dtype):
        assert isinstance(dtype, Type)
        self._dtype = dtype
        name = "dtype(%s)" % (dtype,)
        super(DTypeSpec, self).__init__(name)

    @property
    def key(self):
        return self.dtype

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
    """
    Type class for @jit-compiled functions.
    """

    def __init__(self, dispatcher):
        self._store_object(dispatcher)
        super(Dispatcher, self).__init__("Dispatcher(%s)" % dispatcher)

    def get_call_type(self, context, args, kws):
        """
        Resolve a call to this dispatcher using the given argument types.
        A signature returned and it is ensured that a compiled specialization
        is available for it.
        """
        template, pysig, args, kws = self.dispatcher.get_call_template(args, kws)
        sig = template(context).apply(args, kws)
        sig.pysig = pysig
        return sig

    def get_call_signatures(self):
        sigs = self.dispatcher.nopython_signatures
        return sigs, True

    @property
    def dispatcher(self):
        """
        A strong reference to the underlying numba.dispatcher.Dispatcher instance.
        """
        return self._get_object()

    def get_overload(self, sig):
        """
        Get the compiled overload for the given signature.
        """
        return self.dispatcher.get_overload(sig.args)

    def get_impl_key(self, sig):
        """
        Get the implementation key for the given signature.
        """
        return self.get_overload(sig)


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
    A named native function (resolvable by LLVM) accepting an explicit signature.
    For internal use only.
    """

    def __init__(self, symbol, sig):
        from . import typing
        self.symbol = symbol
        self.sig = sig
        template = typing.make_concrete_template(symbol, symbol, [sig])
        super(ExternalFunction, self).__init__(template)

    @property
    def key(self):
        return self.symbol, self.sig


class NumbaFunction(Function):
    """
    A named native function with the Numba calling convention
    (resolvable by LLVM).
    For internal use only.
    """

    def __init__(self, fndesc, sig):
        from . import typing
        self.fndesc = fndesc
        self.sig = sig
        template = typing.make_concrete_template(fndesc.qualname,
                                                 fndesc.qualname, [sig])
        super(NumbaFunction, self).__init__(template)

    @property
    def key(self):
        return self.fndesc.unique_name, self.sig


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

    def unify(self, typingctx, other):
        if isinstance(other, Pair):
            first = typingctx.unify_pairs(self.first_type, other.first_type)
            second = typingctx.unify_pairs(self.second_type, other.second_type)
            if first != pyobject and second != pyobject:
                return Pair(first, second)


class SimpleIterableType(IterableType):

    def __init__(self, name, iterator_type):
        self._iterator_type = iterator_type
        super(SimpleIterableType, self).__init__(name)

    @property
    def iterator_type(self):
        return self._iterator_type


class SimpleIteratorType(IteratorType):

    def __init__(self, name, yield_type):
        self._yield_type = yield_type
        super(SimpleIteratorType, self).__init__(name)

    @property
    def yield_type(self):
        return self._yield_type


class BaseContainerIterator(SimpleIteratorType):
    """
    Convenience base class for some container iterators.

    Derived classes must implement the *container_class* attribute.
    """

    def __init__(self, container):
        assert isinstance(container, self.container_class), container
        self.container = container
        yield_type = container.dtype
        name = 'iter(%s)' % container
        super(BaseContainerIterator, self).__init__(name, yield_type)

    def unify(self, typingctx, other):
        cls = type(self)
        if isinstance(other, cls):
            container = typingctx.unify_pairs(self.container, other.container)
            if container != pyobject:
                return cls(container)

    @property
    def key(self):
        return self.container


class BaseContainerPayload(Type):
    """
    Convenience base class for some container payloads.

    Derived classes must implement the *container_class* attribute.
    """

    def __init__(self, container):
        assert isinstance(container, self.container_class)
        self.container = container
        name = 'payload(%s)' % container
        super(BaseContainerPayload, self).__init__(name)

    @property
    def key(self):
        return self.container


class RangeType(SimpleIterableType):

    def __init__(self, dtype):
        self.dtype = dtype
        name = "range_state_%s" % (dtype,)
        super(SimpleIterableType, self).__init__(name)
        self._iterator_type = RangeIteratorType(self.dtype)

    def unify(self, typingctx, other):
        if isinstance(other, RangeType):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            if dtype != pyobject:
                return RangeType(dtype)


class RangeIteratorType(SimpleIteratorType):

    def __init__(self, dtype):
        name = "range_iter_%s" % (dtype,)
        super(SimpleIteratorType, self).__init__(name)
        self._yield_type = dtype

    def unify(self, typingctx, other):
        if isinstance(other, RangeIteratorType):
            dtype = typingctx.unify_pairs(self.yield_type, other.yield_type)
            if dtype != pyobject:
                return RangeIteratorType(dtype)


class Generator(SimpleIteratorType):
    """
    Type class for Numba-compiled generator objects.
    """

    def __init__(self, gen_func, yield_type, arg_types, state_types,
                 has_finalizer):
        self.gen_func = gen_func
        self.arg_types = tuple(arg_types)
        self.state_types = tuple(state_types)
        self.has_finalizer = has_finalizer
        name = "%s generator(func=%s, args=%s, has_finalizer=%s)" % (
            yield_type, self.gen_func, self.arg_types,
            self.has_finalizer)
        super(Generator, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.gen_func, self.arg_types, self.yield_type, self.has_finalizer


class NumpyFlatType(SimpleIteratorType, MutableSequence):
    """
    Type class for `ndarray.flat()` objects.
    """

    def __init__(self, arrty):
        self.array_type = arrty
        yield_type = arrty.dtype
        self.dtype = yield_type
        name = "array.flat({arrayty})".format(arrayty=arrty)
        super(NumpyFlatType, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.array_type


class NumpyNdEnumerateType(SimpleIteratorType):
    """
    Type class for `np.ndenumerate()` objects.
    """

    def __init__(self, arrty):
        self.array_type = arrty
        # XXX making this a uintp has the side effect of forcing some
        # arithmetic operations to return a float result.
        yield_type = Tuple((UniTuple(intp, arrty.ndim), arrty.dtype))
        name = "ndenumerate({arrayty})".format(arrayty=arrty)
        super(NumpyNdEnumerateType, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.array_type


class NumpyNdIndexType(SimpleIteratorType):
    """
    Type class for `np.ndindex()` objects.
    """

    def __init__(self, ndim):
        self.ndim = ndim
        yield_type = UniTuple(intp, self.ndim)
        name = "ndindex(dims={ndim})".format(ndim=ndim)
        super(NumpyNdIndexType, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.ndim


class EnumerateType(SimpleIteratorType):
    """
    Type class for `enumerate` objects.
    Type instances are parametered with the underlying source type.
    """

    def __init__(self, iterable_type):
        self.source_type = iterable_type.iterator_type
        yield_type = Tuple([intp, self.source_type.yield_type])
        name = 'enumerate(%s)' % (self.source_type)
        super(EnumerateType, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.source_type


class ZipType(SimpleIteratorType):
    """
    Type class for `zip` objects.
    Type instances are parametered with the underlying source types.
    """

    def __init__(self, iterable_types):
        self.source_types = tuple(tp.iterator_type for tp in iterable_types)
        yield_type = Tuple([tp.yield_type for tp in self.source_types])
        name = 'zip(%s)' % ', '.join(str(tp) for tp in self.source_types)
        super(ZipType, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.source_types


class CharSeq(Type):
    mutable = True

    def __init__(self, count):
        self.count = count
        name = "[char x %d]" % count
        super(CharSeq, self).__init__(name)

    @property
    def key(self):
        return self.count


class UnicodeCharSeq(Type):
    mutable = True

    def __init__(self, count):
        self.count = count
        name = "[unichr x %d]" % count
        super(UnicodeCharSeq, self).__init__(name)

    @property
    def key(self):
        return self.count


class Record(Type):
    """
    A Numpy structured scalar.  *descr* is the string representation
    of the Numpy dtype; *fields* of mapping of field names to
    (type, offset) tuples; *size* the bytesize of a record;
    *aligned* whether the fields are aligned; *dtype* the Numpy dtype
    instance.
    """
    mutable = True

    def __init__(self, descr, fields, size, aligned, dtype):
        self.descr = descr
        self.fields = fields.copy()
        self.size = size
        self.aligned = aligned
        self.dtype = dtype
        name = 'Record(%s)' % descr
        super(Record, self).__init__(name)

    @property
    def key(self):
        # Numpy dtype equality doesn't always succeed, use the descr instead
        # (https://github.com/numpy/numpy/issues/5715)
        return (self.descr, self.size, self.aligned)

    def __len__(self):
        return len(self.fields)

    def offset(self, key):
        return self.fields[key][1]

    def typeof(self, key):
        return self.fields[key][0]

    @property
    def members(self):
        return [(f, t) for f, (t, _) in self.fields.items()]


class ArrayIterator(SimpleIteratorType):
    """
    Type class for iterators of array and buffer objects.
    """

    def __init__(self, array_type):
        self.array_type = array_type
        name = "iter(%s)" % (self.array_type,)
        nd = array_type.ndim
        if nd == 0 or nd == 1:
            yield_type = array_type.dtype
        else:
            yield_type = array_type.copy(ndim=array_type.ndim - 1)
        super(ArrayIterator, self).__init__(name, yield_type)


class Buffer(IterableType, ArrayCompatible):
    """
    Type class for objects providing the buffer protocol.
    Derived classes exist for more specific cases.
    """
    mutable = True
    slice_is_copy = False
    aligned = True

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
        super(Buffer, self).__init__(name)

    @property
    def iterator_type(self):
        return ArrayIterator(self)

    @property
    def as_array(self):
        return self

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

    def __init__(self, dtype, ndim, layout, readonly=False, name=None,
                 aligned=True):
        if readonly:
            self.mutable = False
        if (not aligned or
            (isinstance(dtype, Record) and not dtype.aligned)):
            self.aligned = False
        if name is None:
            type_name = "array"
            if not self.mutable:
                type_name = "readonly " + type_name
            if not self.aligned:
                type_name = "unaligned " + type_name
            name = "%s(%s, %sd, %s)" % (type_name, dtype, ndim, layout)
        super(Array, self).__init__(dtype, ndim, layout, name=name)

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        return Array(dtype=dtype, ndim=ndim, layout=layout, readonly=readonly,
                     aligned=self.aligned)

    @property
    def key(self):
        return self.dtype, self.ndim, self.layout, self.mutable, self.aligned

    def unify(self, typingctx, other):
        """
        Unify this with the *other* Array.
        """
        if (isinstance(other, Array) and other.ndim == self.ndim
            and other.dtype == self.dtype):
            if self.layout == other.layout:
                layout = self.layout
            else:
                layout = 'A'
            readonly = not (self.mutable and other.mutable)
            aligned = self.aligned and other.aligned
            return Array(dtype=self.dtype, ndim=self.ndim, layout=layout,
                         readonly=readonly, aligned=aligned)

    def can_convert_to(self, typingctx, other):
        """
        Convert this Array to the *other*.
        """
        if (isinstance(other, Array) and other.ndim == self.ndim
            and other.dtype == self.dtype):
            if (other.layout in ('A', self.layout)
                and (self.mutable or not other.mutable)
                and (self.aligned or not other.aligned)):
                return Conversion.safe

class SmartArrayType(Array):

    def __init__(self, dtype, ndim, layout, pyclass):
        self.pyclass = pyclass
        super(SmartArrayType, self).__init__(dtype, ndim, layout, name='numba_array')

    @property
    def as_array(self):
        return Array(self.dtype, self.ndim, self.layout)

    def copy(self, dtype=None, ndim=None, layout=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        return type(self)(dtype, ndim, layout, self.pyclass)

class ArrayCTypes(Type):
    """
    This is the type for `numpy.ndarray.ctypes`.
    """
    def __init__(self, arytype):
        # This depends on the ndim for the shape and strides attributes,
        # even though they are not implemented, yet.
        self.ndim = arytype.ndim
        name = "ArrayCTypes(ndim={0})".format(self.ndim)
        super(ArrayCTypes, self).__init__(name)

    @property
    def key(self):
        return self.ndim


class ArrayFlags(Type):
    """
    This is the type for `numpy.ndarray.flags`.
    """
    def __init__(self, arytype):
        self.array_type = arytype
        name = "ArrayFlags({0})".format(self.array_type)
        super(ArrayFlags, self).__init__(name)

    @property
    def key(self):
        return self.array_type


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


class BaseTuple(Hashable):
    """
    The base class for all tuple types (with a known size).
    """

    @classmethod
    def from_types(cls, tys, pyclass=None):
        """
        Instantiate the right tuple type for the given element types.
        """
        homogenous = False
        if tys:
            first = tys[0]
            for ty in tys[1:]:
                if ty != first:
                    break
            else:
                homogenous = True

        if pyclass is not None and pyclass is not tuple:
            # A subclass => is it a namedtuple?
            assert issubclass(pyclass, tuple)
            if hasattr(pyclass, "_asdict"):
                if homogenous:
                    return NamedUniTuple(first, len(tys), pyclass)
                else:
                    return NamedTuple(tys, pyclass)
        if homogenous:
            return UniTuple(first, len(tys))
        else:
            return Tuple(tys)


class BaseAnonymousTuple(BaseTuple):
    """
    Mixin for non-named tuples.
    """

    def can_convert_to(self, typingctx, other):
        """
        Convert this tuple to another one.  Note named tuples are rejected.
        """
        if not isinstance(other, BaseAnonymousTuple):
            return
        if len(self) != len(other):
            return
        if len(self) == 0:
            return Conversion.safe
        if isinstance(other, BaseTuple):
            kinds = [typingctx.can_convert(ta, tb)
                     for ta, tb in zip(self, other)]
            if any(kind is None for kind in kinds):
                return
            return max(kinds)


class _HomogenousTuple(Sequence, BaseTuple):

    @property
    def iterator_type(self):
        return UniTupleIter(self)

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
    def types(self):
        return (self.dtype,) * self.count


class UniTuple(BaseAnonymousTuple, _HomogenousTuple):
    """
    Type class for homogenous tuples.
    """

    def __init__(self, dtype, count):
        self.dtype = dtype
        self.count = count
        name = "(%s x %d)" % (dtype, count)
        super(UniTuple, self).__init__(name)

    @property
    def key(self):
        return self.dtype, self.count

    def unify(self, typingctx, other):
        """
        Unify UniTuples with their dtype
        """
        if isinstance(other, UniTuple) and len(self) == len(other):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            if dtype != pyobject:
                return UniTuple(dtype=dtype, count=self.count)


class UniTupleIter(BaseContainerIterator):
    """
    Type class for homogenous tuple iterators.
    """
    container_class = _HomogenousTuple


class _HeterogenousTuple(BaseTuple):

    def __getitem__(self, i):
        """
        Return element at position i
        """
        return self.types[i]

    def __len__(self):
        # Beware: this makes Tuple(()) false-ish
        return len(self.types)

    def __iter__(self):
        return iter(self.types)


class Tuple(BaseAnonymousTuple, _HeterogenousTuple):

    def __new__(cls, types):
        if types and all(t == types[0] for t in types[1:]):
            return UniTuple(dtype=types[0], count=len(types))
        else:
            return object.__new__(Tuple)

    def __init__(self, types):
        self.types = tuple(types)
        self.count = len(self.types)
        name = "(%s)" % ', '.join(str(i) for i in self.types)
        super(Tuple, self).__init__(name)

    @property
    def key(self):
        return self.types

    def unify(self, typingctx, other):
        """
        Unify elements of Tuples/UniTuples
        """
        # Other is UniTuple or Tuple
        if isinstance(other, BaseTuple) and len(self) == len(other):
            unified = [typingctx.unify_pairs(ta, tb)
                       for ta, tb in zip(self, other)]

            if all(t != pyobject for t in unified):
                return Tuple(unified)


class BaseNamedTuple(BaseTuple):
    pass


class NamedUniTuple(_HomogenousTuple, BaseNamedTuple):

    def __init__(self, dtype, count, cls):
        self.dtype = dtype
        self.count = count
        self.fields = tuple(cls._fields)
        self.instance_class = cls
        name = "%s(%s x %d)" % (cls.__name__, dtype, count)
        super(NamedUniTuple, self).__init__(name)
        self._iterator_type = UniTupleIter(self)

    @property
    def key(self):
        return self.instance_class, self.dtype, self.count


class NamedTuple(_HeterogenousTuple, BaseNamedTuple):

    def __init__(self, types, cls):
        self.types = tuple(types)
        self.count = len(self.types)
        self.fields = tuple(cls._fields)
        self.instance_class = cls
        name = "%s(%s)" % (cls.__name__, ', '.join(str(i) for i in self.types))
        super(NamedTuple, self).__init__(name)

    @property
    def key(self):
        return self.instance_class, self.types


class List(MutableSequence):
    """
    Type class for (arbitrary-sized) homogenous lists.
    """
    mutable = True

    def __init__(self, dtype, reflected=False):
        self.dtype = dtype
        self.reflected = reflected
        cls_name = "reflected list" if reflected else "list"
        name = "%s(%s)" % (cls_name, self.dtype)
        super(List, self).__init__(name=name)

    def copy(self, dtype=None, reflected=None):
        if dtype is None:
            dtype = self.dtype
        if reflected is None:
            reflected = self.reflected
        return List(dtype, reflected)

    def unify(self, typingctx, other):
        if isinstance(other, List):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            reflected = self.reflected or other.reflected
            if dtype != pyobject:
                return List(dtype, reflected)

    @property
    def key(self):
        return self.dtype, self.reflected

    @property
    def iterator_type(self):
        return ListIter(self)

    def is_precise(self):
        return self.dtype.is_precise()


class ListIter(BaseContainerIterator):
    """
    Type class for list iterators.
    """
    container_class = List


class ListPayload(BaseContainerPayload):
    """
    Internal type class for the dynamically-allocated payload of a list.
    """
    container_class = List


class Set(Container):
    """
    Type class for homogenous sets.
    """
    mutable = True

    def __init__(self, dtype):
        assert isinstance(dtype, (Hashable, Undefined))
        self.dtype = dtype
        cls_name = "set"
        name = "%s(%s)" % (cls_name, self.dtype)
        super(Set, self).__init__(name=name)

    @property
    def key(self):
        return self.dtype

    @property
    def iterator_type(self):
        return SetIter(self)

    def is_precise(self):
        return self.dtype.is_precise()

    def copy(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return Set(dtype)

    def unify(self, typingctx, other):
        if isinstance(other, Set):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            if dtype != pyobject:
                return Set(dtype)


class SetIter(BaseContainerIterator):
    """
    Type class for set iterators.
    """
    container_class = Set


class SetPayload(BaseContainerPayload):
    """
    Internal type class for the dynamically-allocated payload of a set.
    """
    container_class = Set


class SetEntry(Type):
    """
    Internal type class for the entries of a Set's hash table.
    """
    def __init__(self, set_type):
        self.set_type = set_type
        name = 'entry(%s)' % set_type
        super(SetEntry, self).__init__(name)

    @property
    def key(self):
        return self.set_type


class MemInfoPointer(Type):
    """
    Pointer to a Numba "meminfo" (i.e. the information for a managed
    piece of memory).
    """
    mutable = True

    def __init__(self, dtype):
        self.dtype = dtype
        name = "memory-managed *%s" % dtype
        super(MemInfoPointer, self).__init__(name)

    @property
    def key(self):
        return self.dtype


class CPointer(Type):
    """
    Type class for pointers to other types.
    """
    mutable = True

    def __init__(self, dtype):
        self.dtype = dtype
        name = "*%s" % dtype
        super(CPointer, self).__init__(name)

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
        super(EphemeralArray, self).__init__(name)

    @property
    def key(self):
        return self.dtype, self.count


class Object(Type):
    mutable = True

    def __init__(self, clsobj):
        self.cls = clsobj
        name = "Object(%s)" % clsobj.__name__
        super(Object, self).__init__(name)

    @property
    def key(self):
        return self.cls


class Optional(Type):
    def __init__(self, typ):
        assert typ != none
        assert not isinstance(typ, (Optional, NoneType))
        self.type = typ
        name = "?%s" % typ
        super(Optional, self).__init__(name)

    @property
    def key(self):
        return self.type

    def can_convert_to(self, typingctx, other):
        if isinstance(other, Optional):
            return typingctx.can_convert(self.type, other.type)
        else:
            conv = typingctx.can_convert(self.type, other)
            if conv is not None:
                return max(conv, Conversion.safe)

    def can_convert_from(self, typingctx, other):
        if other is none:
            return Conversion.promote
        elif isinstance(other, Optional):
            return typingctx.can_convert(other.type, self.type)
        else:
            conv = typingctx.can_convert(other, self.type)
            if conv is not None:
                return max(conv, Conversion.promote)

    def unify(self, typingctx, other):
        if isinstance(other, Optional):
            unified = typingctx.unify_pairs(self.type, other.type)
        else:
            unified = typingctx.unify_pairs(self.type, other)

        if unified != pyobject:
            if isinstance(unified, Optional):
                return unified
            else:
                return Optional(unified)


class NoneType(Opaque):
    """
    The type for None.
    """

    def unify(self, typingctx, other):
        """
        Turn anything to a Optional type;
        """
        if isinstance(other, (Optional, NoneType)):
            return other
        return Optional(other)


class EllipsisType(Opaque):
    """
    The type for the Ellipsis singleton.
    """


class ExceptionClass(Callable, Phantom):
    """
    The type of exception classes (not instances).
    """

    def __init__(self, exc_class):
        assert issubclass(exc_class, BaseException)
        name = "%s" % (exc_class.__name__)
        self.exc_class = exc_class
        super(ExceptionClass, self).__init__(name)

    def get_call_type(self, context, args, kws):
        return self.get_call_signatures()[0][0]

    def get_call_signatures(self):
        from . import typing
        return_type = ExceptionInstance(self.exc_class)
        return [typing.signature(return_type)], False

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
        super(ExceptionInstance, self).__init__(name)

    @property
    def key(self):
        return self.exc_class


class SliceType(Type):

    def __init__(self, name, members):
        assert members in (2, 3)
        self.members = members
        self.has_step = members >= 3
        super(SliceType, self).__init__(name)

    @property
    def key(self):
        return self.members


class ClassInstanceType(Type):
    """
    The type of a jitted class *instance*.  It will be the return-type
    of the constructor of the class.
    """
    mutable = True
    name_prefix = "instance"

    def __init__(self, class_type):
        self.class_type = class_type
        name = "{0}.{1}".format(self.name_prefix, self.class_type.name)
        super(ClassInstanceType, self).__init__(name)

    def get_data_type(self):
        return ClassDataType(self)

    def get_reference_type(self):
        return self

    @property
    def key(self):
        return self.class_type.key

    @property
    def classname(self):
        return self.class_type.class_def.__name__

    @property
    def jitprops(self):
        return self.class_type.jitprops

    @property
    def jitmethods(self):
        return self.class_type.jitmethods

    @property
    def struct(self):
        return self.class_type.struct

    @property
    def methods(self):
        return self.class_type.methods


class ClassType(Callable, Opaque):
    """
    The type of the jitted class (not instance).  When the type of a class
    is called, its constructor is invoked.
    """
    mutable = True
    name_prefix = "jitclass"
    instance_type_class = ClassInstanceType

    def __init__(self, class_def, ctor_template_cls, struct, jitmethods,
                 jitprops):
        self.class_def = class_def
        self.ctor_template = self._specialize_template(ctor_template_cls)
        self.jitmethods = jitmethods
        self.jitprops = jitprops
        self.struct = struct
        self.methods = dict((k, v.py_func) for k, v in self.jitmethods.items())
        fielddesc = ','.join("{0}:{1}".format(k, v) for k, v in struct.items())
        name = "{0}.{1}#{2:x}<{3}>".format(self.name_prefix, class_def.__name__,
                                           id(class_def), fielddesc)
        super(ClassType, self).__init__(name)
        self.instance_type = self.instance_type_class(self)

    def get_call_type(self, context, args, kws):
        return self.ctor_template(context).apply(args, kws)

    def get_call_signatures(self):
        return (), True

    def _specialize_template(self, basecls):
        return type(basecls.__name__, (basecls,), dict(key=self))


class DeferredType(Type):
    """
    Represents a type that will be defined later.  It must be defined
    before it is materialized (used in the compiler).  Once defined, it
    behaves exactly as the type it is defining.
    """
    def __init__(self):
        self._define = None
        name = "{0}#{1}".format(type(self).__name__, id(self))
        super(DeferredType, self).__init__(name)

    def get(self):
        if self._define is None:
            raise RuntimeError("deferred type not defined")
        return self._define

    def define(self, typ):
        if self._define is not None:
            raise TypeError("deferred type already defined")
        if not isinstance(typ, Type):
            raise TypeError("arg is not a Type; got: {0}".format(type(typ)))
        self._define = typ

    def unify(self, typingctx, other):
        return typingctx.unify_pairs(self.get(), other)


class ClassDataType(Type):
    """
    Internal only.
    Represents the data of the instance.  The representation of
    ClassInstanceType contains a pointer to a ClassDataType which represents
    a C structure that contains all the data fields of the class instance.
    """
    def __init__(self, classtyp):
        self.class_type = classtyp
        name = "data.{0}".format(self.class_type.name)
        super(ClassDataType, self).__init__(name)


# Short names

pyobject = PyObject('pyobject')
ffi_forced_object = Opaque('ffi_forced_object')
ffi = Opaque('ffi')
none = NoneType('none')
ellipsis = EllipsisType('...')
Any = Phantom('any')
undefined = Undefined('undefined')
string = Opaque('str')

# No operation is defined on voidptr
# Can only pass it around
voidptr = RawPointer('void*')

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

range_iter32_type = RangeIteratorType(int32)
range_iter64_type = RangeIteratorType(int64)
unsigned_range_iter64_type = RangeIteratorType(uint64)
range_state32_type = RangeType(int32)
range_state64_type = RangeType(int64)
unsigned_range_state64_type = RangeType(uint64)

slice2_type = SliceType('slice<a:b>', 2)
slice3_type = SliceType('slice<a:b:c>', 3)

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

deferred_type = DeferredType

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
ffi
deferred_type
'''.split()
