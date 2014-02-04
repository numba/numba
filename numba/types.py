"""
These type objects do not have a fixed machine representation.  It is up to
the targets to choose their representation.
"""
from __future__ import print_function, division, absolute_import
from collections import defaultdict
import numpy
import ctypes


def _autoincr():
    n = len(_typecache)
    # 4 billion types should be enough, right?
    assert n <= 2 ** 32, "Limited to 4billion types"
    return n

_typecache = defaultdict(_autoincr)


class Type(object):
    __slots__ = '_code', 'name', 'is_parametric'

    def __init__(self, name, param=False):
        self.name = name
        self.is_parametric = param
        self._code = _typecache[self]

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self is other

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


class Integer(Type):
    def cast_python_value(self, value):
        return getattr(numpy, self.name)(value)


class Float(Type):
    def cast_python_value(self, value):
        return getattr(numpy, self.name)(value)


class Complex(Type):
    def cast_python_value(self, value):
        return getattr(numpy, self.name)(value)


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


class Dispatcher(Type):
    def __init__(self, overloaded):
        self.overloaded = overloaded
        super(Dispatcher, self).__init__("Dispatcher(%s)" % overloaded)

    def __eq__(self, other):
        if isinstance(other, Dispatcher):
            return self.overloaded is other.overloaded

    def __hash__(self):
        return hash(self.overloaded)


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


class Array(Type):
    __slots__ = 'dtype', 'ndim', 'layout'

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


class UniTuple(Type):
    def __init__(self, dtype, count):
        self.dtype = dtype
        self.count = count
        name = "(%s x %d)" % (dtype, count)
        super(UniTuple, self).__init__(name, param=True)

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


class UniTupleIter(Type):
    def __init__(self, unituple):
        self.unituple = unituple
        name = 'iter(%s)' % unituple
        super(UniTupleIter, self).__init__(name, param=True)

    def __eq__(self, other):
        if isinstance(other, UniTupleIter):
            return self.unituple == other.unituple

    def __hash__(self):
        return hash(self.unituple)


class Tuple(Type):
    def __init__(self, items):
        self.items = items
        self.count = len(items)
        name = "(%s)" % ', '.join(str(i) for i in items)
        super(Tuple, self).__init__(name, param=True)

    def __getitem__(self, i):
        """
        Return element at position i
        """
        return self.items[i]

    def __len__(self):
        return len(self.items)

    def __eq__(self, other):
        if isinstance(other, Tuple):
            return self.items == other.items

    def __hash__(self):
        return hash(self.items)

    def __iter__(self):
        return iter(self.items)


class CPointer(Type):
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
intp = int32 if tuple.__itemsize__ == 4 else int64
uintp = uint32 if tuple.__itemsize__ == 4 else uint64

float32 = Float('float32')
float64 = Float('float64')

complex64 = Complex('complex64')
complex128 = Complex('complex128')

len_type = Dummy('len')
range_type = Dummy('range')
slice_type = Dummy('slice')
abs_type = Dummy('abs')
print_type = Dummy('print')

range_state32_type = Type('range_state32')
range_state64_type = Type('range_state64')
range_iter32_type = Type('range_iter32')
range_iter64_type = Type('range_iter64')

slice2_type = Type('slice2_type')
slice3_type = Type('slice3_type')

signed_domain = frozenset([int8, int16, int32, int64])
unsigned_domain = frozenset([uint8, uint16, uint32, uint64])
integer_domain = signed_domain | unsigned_domain
real_domain = frozenset([float32, float64])
complex_domain = frozenset([complex64, complex128])
number_domain = real_domain | integer_domain | complex_domain

# Aliases

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

char = globals()["int%d" % (ctypes.sizeof(ctypes.c_char) * 8)]
uchar = byte = globals()["uint%d" % (ctypes.sizeof(ctypes.c_byte) * 8)]
short = globals()["int%d" % (ctypes.sizeof(ctypes.c_short) * 8)]
ushort = globals()["uint%d" % (ctypes.sizeof(ctypes.c_ushort) * 8)]
int_ = globals()["uint%d" % (ctypes.sizeof(ctypes.c_int) * 8)]
uint = globals()["uint%d" % (ctypes.sizeof(ctypes.c_uint) * 8)]
long_ = globals()["int%d" % (ctypes.sizeof(ctypes.c_long) * 8)]
ulong = globals()["uint%d" % (ctypes.sizeof(ctypes.c_ulong) * 8)]
longlong = globals()["int%d" % (ctypes.sizeof(ctypes.c_longlong) * 8)]
ulonglong = globals()["uint%d" % (ctypes.sizeof(ctypes.c_ulonglong) * 8)]


__all__ = '''
int8
int16
int32
int64
uint8
uint16
uint32
uint64
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
