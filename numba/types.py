"""
Type objects that do not have a fixed machine representation.  It is up to
the targets to choose their representation.

Classes
    Type, base class for numba's type system. 
        Integer, the integer type
        Float, floating-point type
        Complex, comple number type
        Dummy, types that defy easy categorization

"""
from __future__ import print_function, division, absolute_import
from collections import defaultdict

import numpy
import weakref

from . import utils


def _autoincr():
    n = len(_typecache)
    # 4 billion types should be enough, right?
    assert n <= 2 ** 32, "Limited to 4 billion types"
    return n


_typecache = defaultdict(_autoincr)


class Type(object):
    """
    The base class of all numba types.

    The default behavior is to provide equality through `name` attribute.
    Two types are equal if there `name` are equal.
    Subclass can refine this behavior.

    A number of instances are predefined:

    Booleans, evaluate to True or False
    ---------------
    boolean
    b1, same as numpy boolean.
    bool_, same as numpy boolean.
    """
    __slots__ = '_code', 'name', 'is_parametric'

    mutable = False

    def __init__(self, name, param=False):
        self.name = name
        self.is_parametric = param
        self._code = _typecache[self]

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.name == other.name

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


class OpaqueType(Type):
    """
    To deal with externally defined literal types
    """

    def __init__(self, name):
        super(OpaqueType, self).__init__(name)


@utils.total_ordering
class Integer(Type):
    """
    Defines an Integer type to a specific bit width.

    A number of Intger types (instances) are already defined for convenience. 
    Integer types are defined in terms of bits or in terms of bytes as in numpy.
    
    unsigned types, bit versions
    ----------------
    uint8, an 8-bit unsigned integer
    uint16, a 16-bit unsigned integer
    uint32, a 32-bit unsigned integer
    uint64, a 64-bit unsigned integer
    
    signed types, bit versions
    ----------------
    int8, an 8-bit signed ingeger
    int16, a 16-bit signed integer
    int32, a 32-bit signed integer
    int64, a 64-bit signed integer

    unsigned types, byte versions
    ----------------
    u1, a 1-byte unsigned integer
    u2, a 2-byte unsigned integer
    u4, a 4-byte unsigned integer
    u8, a 8-byte unsigned integer
    
    signed types, byte versions
    ----------------
    i1, a 1-byte signed ingeger
    i2, a 2-byte signed integer
    i4, a 4-byte signed integer
    i8, a 8-byte signed integer

    convenience types
    These types automatically use the width of the machine.
    ----------------
    intp, signed 32- or 64-bit integer depending on the machine width 
    uintp, unsigned 32- or 64-bit integer depending on the machine width

    numpy integer types
    These types create numpy dtypes. They all take a parmeter n. For example, b = char(17)
    ----------------
    byte(n), an unsigned numpy byte (8-bits).
    char(n), signed numpy byte (8-bits). 
    uchar(n), an unsigned numpy byte (8-bits).
    short(n), signed numpy short (16-bits).
    ushort(n), unsigned numpy short (16-bits).
    int_(n), signed C long, either 32- or 64-bit wide depending on the machine.
    intc(n), signed C int (32-bits).
    uintc(n), unsigned C int (32-bits)  
    long_(n), signed numpy long, size depends on machine width
    ulong(n), unsigned numpy long, size depends on machine width
    longlong(n), signed numpy longlong, size depends on machine width
    ulonglong(n), unsigned numpy longlong, size depends on machine width
    """
    
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
class Float(Type):
    """
    Defines Floating Point types.

    For convenience, a number of instances have already been defined.

    32 bit floating point types
    Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    -----------------------
    f4, a 4-byte float
    float32, a 32-bit float
    float_, a 4-byte float

    64 bit precision floating point types
    Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    -----------------------
    float64, a 64-bit float
    f8, an 8-byte float

    """

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
class Complex(Type):
    """
    Defines Complex types a + bi (a + bj)

    real and imaginary types are 32-bit, single precision values
    ----------------
    complex6i4, two 32-bit floats
    c8, two 4-byte floats

    real and imaginary types are 64-bit, double precision values
    ----------------
    complex128, two 64-bit floats
    c16, two 8 byte floats

    """

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


class Prototype(Type):
    def __init__(self, args, return_type):
        self.args = args
        self.return_type = return_type
        name = "%s(%s)" % (return_type, ', '.join(str(a) for a in args))
        super(Prototype, self).__init__(name=name)


class Dummy(Type):
    """
    For types that do not really have a representation and are compatible
    with void *.

    For convenience, a number of Dummy types (instances) have already been defined:
    ------------------
    none
    void
    """


class Kind(Type):
    def __init__(self, of):
        self.of = of
        super(Kind, self).__init__("kind(%s)" % of)

    def __eq__(self, other):
        if isinstance(other, Kind):
            return self.of == other.of

    def __hash__(self):
        return hash(self.of)


class Module(Type):
    def __init__(self, pymod):
        self.pymod = pymod
        super(Module, self).__init__("Module(%s)" % pymod)

    def __eq__(self, other):
        if isinstance(other, Module):
            return self.pymod == other.pymod

    def __hash__(self):
        return hash(self.pymod)


class Macro(Type):
    def __init__(self, template):
        self.template = template
        cls = type(self)
        super(Macro, self).__init__("%s(%s)" % (cls.__name__, template))

    def __eq__(self, other):
        if isinstance(other, Macro):
            return self.template == other.template

    def __hash__(self):
        # FIXME maybe this should not be hashable
        return hash(self.template)


class Function(Type):
    def __init__(self, template):
        self.template = template
        cls = type(self)
        # TODO template is mutable.  Should use different naming scheme
        super(Function, self).__init__("%s(%s)" % (cls.__name__, template))

    def __eq__(self, other):
        if isinstance(other, Function):
            return self.template == other.template

    def __hash__(self):
        # FIXME maybe this should not be hashable
        return hash(self.template)

    def extend(self, template):
        self.template.cases.extend(template.cases)


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

    def __eq__(self, other):
        if type(self) is type(other):
            obj = self._wr()
            return obj is not None and obj is other._wr()

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self._wr)


class Dispatcher(WeakType):

    def __init__(self, overloaded):
        self._store_object(overloaded)
        super(Dispatcher, self).__init__("Dispatcher(%s)" % overloaded)

    @property
    def overloaded(self):
        """
        A strong reference to the underlying Dispatcher instance.
        """
        return self._get_object()


class FunctionPointer(Function):
    def __init__(self, template, funcptr):
        self.funcptr = funcptr
        super(FunctionPointer, self).__init__(template)


class Method(Function):
    def __init__(self, template, this):
        self.this = this
        newcls = type(template.__name__ + '.' + str(this), (template,),
                      dict(this=this))
        super(Method, self).__init__(newcls)

    def __eq__(self, other):
        if isinstance(other, Method):
            return (self.template.__name__ == other.template.__name__ and
                    self.this == other.this)

    def __hash__(self):
        return hash((self.template.__name__, self.this))


class Pair(Type):
    """
    A heterogenous pair.
    """

    def __init__(self, first_type, second_type):
        self.first_type = first_type
        self.second_type = second_type
        name = "pair<%s, %s>" % (first_type, second_type)
        super(Pair, self).__init__(name=name)

    def __eq__(self, other):
        if isinstance(other, Pair):
            return (self.first_type == other.first_type and
                    self.second_type == other.second_type)

    def __hash__(self):
        return hash((self.first_type, self.second_type))


class IterableType(Type):
    """
    Base class for iterable types.
    Derived classes should implement the *iterator_type* attribute.
    """


class SimpleIterableType(IterableType):

    def __init__(self, name, iterator_type):
        self.iterator_type = iterator_type
        super(SimpleIterableType, self).__init__(name, param=True)

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return self.name == other.name

    def __hash__(self):
        return hash(self.name)


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

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class RangeType(SimpleIterableType):
    pass

class RangeIteratorType(SimpleIteratorType):
    pass


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

    def __eq__(self, other):
        if isinstance(other, EnumerateType):
            return self.source_type == other.source_type

    def __hash__(self):
        return hash(self.source_type)


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

    def __eq__(self, other):
        if isinstance(other, ZipType):
            return self.source_types == other.source_types

    def __hash__(self):
        return hash(self.source_types)


class CharSeq(Type):
    mutable = True

    def __init__(self, count):
        self.count = count
        name = "[char x %d]" % count
        super(CharSeq, self).__init__(name, param=True)

    def __eq__(self, other):
        if isinstance(other, CharSeq):
            return self.count == other.count

    def __hash__(self):
        return hash(self.name)


class UnicodeCharSeq(Type):
    mutable = True

    def __init__(self, count):
        self.count = count
        name = "[unichr x %d]" % count
        super(UnicodeCharSeq, self).__init__(name, param=True)

    def __eq__(self, other):
        if isinstance(other, UnicodeCharSeq):
            return self.count == other.count

    def __hash__(self):
        return hash(self.name)


class Record(Type):
    mutable = True

    def __init__(self, id, fields, size, align, dtype):
        self.id = id
        self.fields = fields.copy()
        self.size = size
        self.align = align
        self.dtype = dtype
        name = 'Record(%s)' % id
        super(Record, self).__init__(name)

    def __eq__(self, other):
        if isinstance(other, Record):
            return (self.id == other.id and
                    self.size == other.size and
                    self.align == other.align)

    def __hash__(self):
        return hash(self.name)

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

    def __init__(self, array_type):
        self.array_type = array_type
        name = "iter(%s)" % (self.array_type,)
        if array_type.ndim == 1:
            self.yield_type = array_type.dtype
        else:
            self.yield_type = array_type.copy(ndim=array_type.ndim - 1)
        super(ArrayIterator, self).__init__(name, param=True)


class Array(IterableType):
    __slots__ = 'dtype', 'ndim', 'layout'

    mutable = True

    # CS and FS are not reserved for inner contig but strided
    LAYOUTS = frozenset(['C', 'F', 'CS', 'FS', 'A'])

    def __init__(self, dtype, ndim, layout):
        from numba.typeconv.rules import default_type_manager as tm

        if isinstance(dtype, Array):
            raise TypeError("Array dtype cannot be Array")
        if layout not in self.LAYOUTS:
            raise ValueError("Invalid layout '%s'" % layout)

        self.dtype = dtype
        self.ndim = ndim
        self.layout = layout
        name = "array(%s, %sd, %s)" % (dtype, ndim, layout)
        super(Array, self).__init__(name, param=True)
        self.iterator_type = ArrayIterator(self)

        if layout != 'A':
            # Install conversion from non-any layout to any layout
            ary_any = Array(dtype, ndim, 'A')
            tm.set_safe_convert(self, ary_any)

    def copy(self, dtype=None, ndim=None, layout=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        return Array(dtype=dtype, ndim=ndim, layout=layout)

    def get_layout(self, dim):
        assert 0 <= dim < self.ndim
        if self.layout in 'CFA':
            return self.layout
        elif self.layout == 'CS':
            if dim == self.ndim - 1:
                return 'C'
        elif self.layout == 'FS':
            if dim == 0:
                return 'F'
        return 'A'

    def getitem(self, ind):
        """Returns (return-type, index-type)
        """
        if isinstance(ind, UniTuple):
            idxty = UniTuple(intp, ind.count)
        else:
            idxty = intp
        return self.dtype, idxty

    def setitem(self):
        """Returns (index-type, value-type)
        """
        return intp, self.dtype

    def __eq__(self, other):
        if isinstance(other, Array):
            return (self.dtype == other.dtype and
                    self.ndim == other.ndim and
                    self.layout == other.layout)

    def __hash__(self):
        return hash((self.dtype, self.ndim, self.layout))


    @property
    def is_c_contig(self):
        return self.layout == 'C' or (self.ndim == 1 and self.layout in 'CF')

    @property
    def is_f_contig(self):
        return self.layout == 'F' or (self.ndim == 1 and self.layout in 'CF')

    @property
    def is_contig(self):
        return self.layout in 'CF'


class UniTuple(IterableType):

    def __init__(self, dtype, count):
        self.dtype = dtype
        self.count = count
        name = "(%s x %d)" % (dtype, count)
        super(UniTuple, self).__init__(name, param=True)
        self.iterator_type = UniTupleIter(self)

    def getitem(self, ind):
        if isinstance(ind, UniTuple):
            idxty = UniTuple(intp, ind.count)
        else:
            idxty = intp
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

    def __eq__(self, other):
        if isinstance(other, UniTuple):
            return self.dtype == other.dtype and self.count == other.count

    def __hash__(self):
        return hash((self.dtype, self.count))


class UniTupleIter(IteratorType):

    def __init__(self, unituple):
        self.unituple = unituple
        self.yield_type = unituple.dtype
        name = 'iter(%s)' % unituple
        super(UniTupleIter, self).__init__(name, param=True)

    def __eq__(self, other):
        if isinstance(other, UniTupleIter):
            return self.unituple == other.unituple

    def __hash__(self):
        return hash(self.unituple)


class Tuple(Type):
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
        return len(self.types)

    def __eq__(self, other):
        if isinstance(other, Tuple):
            return self.types == other.types

    def __hash__(self):
        return hash(self.types)

    def __iter__(self):
        return iter(self.types)


class CPointer(Type):
    mutable = True

    def __init__(self, dtype):
        self.dtype = dtype
        name = "*%s" % dtype
        super(CPointer, self).__init__(name, param=True)

    def __eq__(self, other):
        if isinstance(other, CPointer):
            return self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype)


class Object(Type):
    mutable = True

    def __init__(self, clsobj):
        self.cls = clsobj
        name = "Object(%s)" % clsobj.__name__
        super(Object, self).__init__(name, param=True)

    def __eq__(self, other):
        if isinstance(other, Object):
            return self.cls == other.cls

    def __hash__(self):
        return hash(self.cls)


class Optional(Type):
    def __init__(self, typ):
        self.type = typ
        name = "?%s" % typ
        super(Optional, self).__init__(name, param=True)

    def __eq__(self, other):
        if isinstance(other, Optional):
            return self.type == other.type

    def __hash__(self):
        return hash(self.type)


# Utils

def is_int_tuple(x):
    if isinstance(x, Tuple):
        return all(i in integer_domain for i in x.types)
    elif isinstance(x, UniTuple):
        return x.dtype in integer_domain
    else:
        return False

# Short names


pyobject = Type('pyobject')
none = Dummy('none')
Any = Dummy('any')
VarArg = Dummy('...')
string = Dummy('str')

# No operation is defined on voidptr
# Can only pass it around
voidptr = Dummy('void*')

boolean = bool_ = Type('bool')

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

float32 = Float('float32')
float64 = Float('float64')

complex64 = Complex('complex64', float32)
complex128 = Complex('complex128', float64)

len_type = Dummy('len')
range_type = Dummy('range')
slice_type = Dummy('slice')
abs_type = Dummy('abs')
neg_type = Dummy('neg')
print_type = Dummy('print')
print_item_type = Dummy('print-item')
sign_type = Dummy('sign')
exception_type = Dummy('exception')

range_iter32_type = RangeIteratorType('range_iter32', int32)
range_iter64_type = RangeIteratorType('range_iter64', int64)
range_state32_type = RangeType('range_state32', range_iter32_type)
range_state64_type = RangeType('range_state64', range_iter64_type)

# slice2_type = Type('slice2_type')
slice3_type = Type('slice3_type')

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
intc
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
'''.split()
