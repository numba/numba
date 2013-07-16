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

    def ctype_as_argument(self):
        if not hasattr(self.desc, 'ctype_as_argument'):
            raise TypeError('%s cannot be used as ctype argument' % self)
        else:
            return self.desc.ctype_as_argument()

    def ctype_as_return(self):
        if not hasattr(self.desc, 'ctype_as_return'):
            return self.desc.ctype_as_argument()
        else:
            return self.desc.ctype_as_return()

    def ctype_pack_argument(self, value):
        if not hasattr(self.desc, 'ctype_pack_argument'):
            cty = self.ctype_as_argument()
            return cty(value)
        else:
            return self.desc.ctype_pack_argument(value)

    def ctype_unpack_return(self, value):
        if not hasattr(self.desc, 'ctype_unpack_return'):
            return value.value
        else:
            return self.desc.ctype_unpack_return(value)

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
            if self.bitwidth < 32:
                return other.bitwidth - 32 + 50
            else:
                return 64 - other.bitwidth + 50
        elif isinstance(other, Complex):
            if self.bitwidth < 32:
                return other.bitwidth//2 - 32 + 100
            else:
                return 64 - other.bitwidth//2 + 100

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
        elif isinstance(dst, Float):
            return builder.sitofp(value, dst.llvm_as_value())

    def ctype_as_argument(self):
        return getattr(ct, 'c_int%d' % self.bitwidth)

class Unsigned(Integer):
    def __init__(self, bitwidth):
        super(Unsigned, self).__init__(False, bitwidth)

    def llvm_const(self, builder, value):
        return lc.Constant.int(self.llvm_as_value(), value)
    
    def llvm_cast(self, builder, value, dst):
        if isinstance(dst, Integer):
            if dst.bitwidth > self.bitwidth:
                return builder.zext(value, dst.llvm_as_value())
            else:
                return builder.trunc(value, dst.llvm_as_value())
        elif isinstance(dst, Float):
            return builder.uitofp(value, dst.llvm_as_value())

    def ctype_as_argument(self):
        return getattr(ct, 'c_uint%d' % self.bitwidth)

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

    def llvm_as_value(self):
        return {32: lc.Type.float(), 64: lc.Type.double()}[self.bitwidth]

    def llvm_const(self, builder, value):
        return lc.Constant.real(self.llvm_as_value(), value)

    def llvm_cast(self, builder, value, dst):
        if isinstance(dst, Integer):
            op = (builder.fptosi if dst.signed else builder.fptoui)
            return op(value, dst.llvm_as_value())
        elif isinstance(dst, Float):
            if dst.bitwidth > self.bitwidth:
                return builder.fpext(value, dst.llvm_as_value())
            else:
                return builder.fptrunc(value, dst.llvm_as_value())

    def ctype_as_argument(self):
        ctname = {32: 'c_float', 64: 'c_double'}[self.bitwidth]
        return getattr(ct, ctname)


class c_complex_base(ct.Structure):
    def __init__(self, real=0, imag=0):
        if isinstance(real, complex):
            real, imag = real.real, real.imag
        self.real = real
        self.imag = imag

class c_complex64(c_complex_base):
    _fields_ = [('real', ct.c_float),
                ('imag', ct.c_float),]

class c_complex128(c_complex_base):
    _fields_ = [('real', ct.c_double),
                ('imag', ct.c_double),]


class Complex(object):
    fields = 'bitwidth',
    
    def __init__(self, bitwidth):
        self.bitwidth = bitwidth
        self.element = Type(Float(self.bitwidth // 2))

    def coerce(self, other):
        if isinstance(other, Complex):
            return (other.bitwidth - self.bitwidth) // 8

    def __repr__(self):
        return 'complex%d' % self.bitwidth

    def llvm_as_value(self):
        return lc.Type.struct([self.element.llvm_as_value()] * 2)

    def llvm_as_argument(self):
        return lc.Type.pointer(self.llvm_as_value())

    def llvm_const(self, builder, value):
        real = self.element.llvm_const(builder, value.real)
        imag = self.element.llvm_const(builder, value.imag)
        return self.llvm_pack(builder, real, imag)

    def llvm_value_from_arg(self, builder, value):
        return builder.load(value)

    def ctype_value(self):
        return {64: c_complex64, 128: c_complex128}[self.bitwidth]

    def ctype_as_argument(self):
        return ct.POINTER(self.ctype_value())

    def ctype_as_return(self):
        return self.ctype_value()

    def ctype_pack_argument(self, value):
        cty = self.ctype_value()
        val = cty(value)
        return ct.byref(val)

    def ctype_unpack_return(self, value):
        return complex(value.real, value.imag)

    def llvm_pack(self, builder, real, imag):
        c = lc.Constant.undef(self.llvm_as_value())
        c = builder.insert_value(c, real, 0)
        c = builder.insert_value(c, imag, 1)
        return c

    def llvm_unpack(self, builder, value):
        real = builder.extract_value(value, 0)
        imag = builder.extract_value(value, 1)
        return real, imag

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
