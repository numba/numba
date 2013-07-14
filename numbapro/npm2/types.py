from math import copysign

__all__ = ('int8', 'int16', 'int32', 'int64', 'intp',
           'uint8', 'uint16', 'uint32', 'uint64',
           'float32', 'float64', 'complex64', 'complex128',
           'arraytype')

class Type(object):
    def __new__(cls, desc):
        if isinstance(desc, Type):
            return desc
        else:
            cls.check_interface(desc)
            instance = object.__new__(cls)
            instance.desc = desc
            return instance

    @classmethod
    def check_interface(cls, desc):
        assert hasattr(desc, 'coerce')
        assert hasattr(desc, 'fields')

    def coerce(self, other):
        ret = self.try_coerce(other)
        if ret is None:
            raise TypeError('can not coerce %s -> %s' % (other, self))
        return ret

    def try_coerce(self, other):
        return self.desc.coerce(Type(other).desc)


    def __str__(self):
        return str(self.desc)

    def __repr__(self):
        return repr(self.desc)

    def list_fields(self):
        values = []
        for field in self.desc.fields:
            values.append(getattr(self.desc, field))
        return values

    def __hash__(self):
        return hash(tuple(self.list_fields()))

    def __eq__(self, other):
        if type(self.desc) == type(other.desc):
            return all(a == b
                        for a, b
                        in zip(self.list_fields(), Type(other).list_fields()))

class Integer(object):
    fields = 'signed', 'bitwidth'
    
    def __init__(self, signed, bitwidth):
        self.signed = signed
        self.bitwidth = bitwidth

    def __repr__(self):
        if self.signed:
            return 'int%d' % self.bitwidth
        else:
            return 'uint%d' % self.bitwidth

    def coerce(self, other):
        if isinstance(other, Integer):
            pts = (other.bitwidth - self.bitwidth) // 8
            if self.signed != other.signed: # signedness mismatch
                pts += copysign(25, pts)
            return pts
        elif isinstance(other, Float):
            return (other.bitwidth - self.bitwidth) // 8 + 50
        elif isinstance(other, Complex):
            return (other.bitwidth - self.bitwidth) // 8 + 100


class Signed(Integer):
    def __init__(self, bitwidth):
        super(Signed, self).__init__(True, bitwidth)

class Unsigned(Integer):
    def __init__(self, bitwidth):
        super(Unsigned, self).__init__(False, bitwidth)


class Float(object):
    fields = 'bitwidth',

    def __init__(self, bitwidth):
        self.bitwidth = bitwidth

    def coerce(self, other):
        if isinstance(other, Float):
            return (other.bitwidth - self.bitwidth) // 8
        elif isinstance(other, Integer):
            factor = 0 if other.signed else 1
            return (other.bitwidth - self.bitwidth) // 8 - 50 - 1
        elif isinstance(other, Complex):
            return (other.bitwidth - self.bitwidth) // 8 + 50

    def __repr__(self):
        return 'float%d' % self.bitwidth


class Complex(object):
    fields = 'bitwidth',
    
    def __init__(self, bitwidth):
        self.bitwidth = bitwidth

    def coerce(self, other):
        if isinstance(other, Complex):
            return (other.bitwidth - self.bitwidth) // 8

    def __repr__(self):
        return 'complex%d' % self.bitwidth


class Array(object):
    fields = 'element', 'shape', 'layout'
    
    def __init__(self, element, shape, layout):
        assert isinstance(shape, tuple)
        assert layout in 'CFA'
        self.shape = shape
        self.element = element
        self.layout = layout

    def coerce(self, other):
        return (self == other)

    def __repr__(self):
        return 'complex%d' % self.bitwidth


class BuiltinObject(object):
    fields = 'name',
    
    def __init__(self, name):
        self.name = name

    def coerce(self, other):
        if isinstance(other, BuiltinObject):
            return self.name == other.name

    def __repr__(self):
        return '<builtin %s>' % self.name

module_type = Type(BuiltinObject('module'))
function_type = Type(BuiltinObject('function'))
range_type = Type(BuiltinObject('range'))
range_iter_type = Type(BuiltinObject('range-iter'))

void = Type(BuiltinObject('void'))

boolean = Type(Unsigned(1))

int8  = Type(Signed(8))
int16 = Type(Signed(16))
int32 = Type(Signed(32))
int64 = Type(Signed(64))

uint8  = Type(Unsigned(8))
uint16 = Type(Unsigned(16))
uint32 = Type(Unsigned(32))
uint64 = Type(Unsigned(64))

float32 = Type(Float(32))
float64 = Type(Float(64))

complex64 = Type(Complex(64))
complex128 = Type(Complex(128))

def arraytype(element, shape, layout):
    return Type(Array(element, shape, layout))

intp = {4: int32, 8: int64}[tuple.__itemsize__]
