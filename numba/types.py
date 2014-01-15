"""
These type objects do not have a fixed machine representation.  It is up to
the targets to choose their representation.
"""
from __future__ import print_function, division, absolute_import


class Type(object):
    is_parametric = False

    def __init__(self, name, param=False):
        self.name = name
        self.is_parametric = param

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return not (self == other)

    def __call__(self, *args):
        return Prototype(args=args, return_type=self)

    def __getitem__(self, args):
        assert not isinstance(self, Array)
        if isinstance(args, tuple):
            ndim = len(args)
            if args[0].step == 1:
                layout = 'F'
            elif args[-1].step == 1:
                layout = 'C'
            else:
                layout = 'A'
        else:
            ndim = 1
            layout = 'A'

        return Array(dtype=self, ndim=ndim, layout=layout)

    __iter__ = NotImplemented


class Prototype(Type):
    def __init__(self, args, return_type):
        name = "%s%s" % (return_type, ', '.join(str(a) for a in args))
        super(Prototype, self).__init__(name=name)
        self.args = args
        self.return_type = return_type


class Dummy(Type):
    """
    For type that does not really have a representation.
    """


class Kind(Type):
    def __init__(self, of):
        super(Kind, self).__init__("kind(%s)" % of)
        self.of = of

    def __eq__(self, other):
        if isinstance(other, Kind):
            return self.of == other.of

    def __hash__(self):
        return hash(self.of)


class Module(Type):
    def __init__(self, pymod):
        super(Module, self).__init__("Module(%s)" % pymod)
        self.pymod = pymod

    def __eq__(self, other):
        if isinstance(other, Module):
            return self.pymod == other.pymod

    def __hash__(self):
        return hash(self.pymod)


class Function(Type):
    def __init__(self, template):
        cls = type(self)
        super(Function, self).__init__("%s(%s)" % (cls.__name__, template))
        self.template = template

    def __eq__(self, other):
        if isinstance(other, Function):
            return self.template == other.template

    def __hash__(self):
        return hash(self.template)


class Method(Function):
    def __init__(self, template, this):
        newcls = type(template.__name__ + '.' + str(this), (template,),
                      dict(this=this))
        super(Method, self).__init__(newcls)
        self.this = this

    def __eq__(self, other):
        if isinstance(other, Method):
            return (self.template.__name__ == other.template.__name__ and
                    self.this == other.this)

    def __hash__(self):
        return hash((self.template.__name__, self.this))


class Array(Type):
    LAYOUTS = frozenset(['C', 'F', 'CS', 'FS', 'A'])

    def __init__(self, dtype, ndim, layout):
        if isinstance(dtype, Array):
            raise TypeError("Array dtype cannot be Array")
        if layout not in self.LAYOUTS:
            raise ValueError("Invalid layout '%s'" % layout)
        name = "array(%s, %sd, %s)" % (dtype, ndim, layout)
        super(Array, self).__init__(name, param=True)
        self.dtype = dtype
        self.ndim = ndim
        self.layout = layout

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


class UniTuple(Type):
    def __init__(self, dtype, count):
        name = "(%s x %d)" % (dtype, count)
        super(UniTuple, self).__init__(name, param=True)
        self.dtype = dtype
        self.count = count

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
        name = 'iter(%s)' % unituple
        super(UniTupleIter, self).__init__(name, param=True)
        self.unituple = unituple

    def __eq__(self, other):
        if isinstance(other, UniTupleIter):
            return self.unituple == other.unituple

    def __hash__(self):
        return hash(self.unituple)


class Tuple(Type):
    def __init__(self, items):
        name = "(%s)" % ', '.join(str(i) for i in items)
        super(Tuple, self).__init__(name, param=True)
        self.items = items

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


class CPointer(Type):
    def __init__(self, dtype):
        name = "*%s" % dtype
        super(CPointer, self).__init__(name, param=True)
        self.dtype = dtype

    def __eq__(self, other):
        if isinstance(other, CPointer):
            return self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype)


class Object(Type):
    def __init__(self, clsobj):
        name = "Object(%s)" % clsobj.__name__
        super(Object, self).__init__(name, param=True)
        self.cls = clsobj

    def __eq__(self, other):
        if isinstance(other, Object):
            return self.cls == other.cls

    def __hash__(self):
        return hash(self.cls)


class Optional(Type):
    def __init__(self, typ):
        name = "?%s" % typ
        super(Optional, self).__init__(name, param=True)
        self.type = typ

    def __eq__(self, other):
        if isinstance(other, Optional):
            return self.type == other.type

    def __hash__(self):
        return hash(self.type)


pyobject = Type('pyobject')
none = Dummy('none')
Any = Dummy('any')
string = Dummy('str')

boolean = bool_ = Type('bool')

byte = uint8 = Type('uint8')
uint16 = Type('uint16')
uint32 = Type('uint32')
uint64 = Type('uint64')

int8 = Type('int8')
int16 = Type('int16')
int32 = Type('int32')
int64 = Type('int64')
intp = int32 if tuple.__itemsize__ == 4 else int64

float32 = Type('float32')
float64 = Type('float64')

complex64 = Type('complex64')
complex128 = Type('complex128')

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

domains = unsigned_domain, signed_domain, real_domain
