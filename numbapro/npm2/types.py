from math import copysign
from llvm import core as lc
import ctypes as ct

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

    def llvm_as_value(self):
        if not hasattr(self.desc, 'llvm_as_value'):
            raise TypeError('%s cannot be used as value' % self)
        else:
            return self.desc.llvm_as_value()

    def llvm_as_argument(self):
        if not hasattr(self.desc, 'llvm_as_argument'):
            ret = self.llvm_as_value()
        else:
            ret = self.desc.llvm_as_argument()
        if ret is None:
            raise TypeError('%s cannot be used as argument type')
        return ret

    def llvm_as_return(self):
        if not hasattr(self.desc, 'llvm_as_return'):
            ret = lc.Type.pointer(self.llvm_as_value())
        else:
            ret = self.desc.llvm_as_return()
        if ret is None:
            raise TypeError('%s cannot be used as return type')
        return ret

    def llvm_value_from_arg(self, builder, arg):
        if not hasattr(self.desc, 'llvm_value_from_arg'):
            ret = arg
        else:
            ret = self.desc.llvm_value_from_arg(builder, arg)
        return ret

    def llvm_cast(self, builder, val, dst):
        if not hasattr(self.desc, 'llvm_cast'):
            raise TypeError('%s does not support casting' % self)
        ret = self.desc.llvm_cast(builder, val, dst.desc)
        if ret is None:
            raise TypeError('%s cannot be casted to %s' % (self, dst))
        return ret

    def llvm_const(self, builder, value):
        if not hasattr(self.desc, 'llvm_const'):
            raise TypeError('%s does not support constant value' % self)
        else:
            return self.desc.llvm_const(builder, value)

    def ctype_argument(self):
        if not hasattr(self.desc, 'ctype_argument'):
            raise TypeError('%s cannot be used as ctype argument' % self)
        else:
            return self.desc.ctype_argument()

    def ctype_return(self):
        if not hasattr(self.desc, 'ctype_return'):
            raise TypeError('%s cannot be used as ctype return' % self)
        else:
            return self.desc.ctype_return()

    def ctype_pack_argument(self, value):
        if not hasattr(self.desc, 'ctype_pack_argument'):
            return self.ctype_argument()(value)
        else:
            return self.desc.ctype_prepare_argument(value)

    def ctype_unpack_return(self, value):
        if not hasattr(self.desc, 'ctype_unpack_return'):
            return value.value
        else:
            return self.desc.ctype_prepare_return(value)

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

    def llvm_as_value(self):
        return lc.Type.int(self.bitwidth)

class Signed(Integer):
    def __init__(self, bitwidth):
        super(Signed, self).__init__(True, bitwidth)

    def llvm_const(self, builder, value):
        return lc.Constant.int_signextend(self.llvm_as_value(), value)

    def llvm_cast(self, builder, value, dst):
        if isinstance(dst, Integer):
            if dst.bitwidth > self.bitwidth:
                return builder.sext(value, dst.llvm_as_value())
            else:
                return builder.trunc(value, dst.llvm_as_value())

    def ctype_argument(self):
        return getattr(ct, 'c_int%d' % self.bitwidth)

    def ctype_return(self):
        return self.ctype_argument()


class Unsigned(Integer):
    def __init__(self, bitwidth):
        super(Unsigned, self).__init__(False, bitwidth)

    def llvm_const(self, builder, value):
        return lc.Constant.int(self.llvm_as_value(), value)

    def ctype_argument(self):
        return getattr(ct, 'c_uint%d' % self.bitwidth)

    def ctype_return(self):
        return self.ctype_argument()

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

class RangeType(BuiltinObject):
    def llvm_as_value(self):
        elem = intp.llvm_as_value()
        return lc.Type.struct((elem, elem, elem))

range_type = Type(RangeType('range'))
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
