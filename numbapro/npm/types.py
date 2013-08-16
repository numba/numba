from math import copysign
from llvm import core as lc
import ctypes as ct
from . import cgutils

__all__ = ('int8', 'int16', 'int32', 'int64', 'intp',
           'uint8', 'uint16', 'uint32', 'uint64',
           'float32', 'float64', 'complex64', 'complex128',
           'arraytype')

ERRMSG_CHANGE_OF_SIGN = 'cast resulting in change of sign'
ERRMSG_NEGATIVE_TO_UNSIGNED = 'casting negative value to unsigned'
ERRMSG_NAN = 'casting NaN to integer'

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
        assert hasattr(desc, 'coerce'), ('%s must define "coerce"' % cls)
        assert hasattr(desc, 'fields'), ('%s must define "fields"' % cls)

    def coerce(self, other):
        ret = self.try_coerce(other)
        if ret is None:
            raise TypeError('can not coerce %s -> %s' % (other, self))
        return ret

    def unpack(self, count):
        if hasattr(self.desc, 'unpack'):
            return self.desc.unpack(count)
        else:
            raise TypeError('can not unpack %s' % (self))

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
            raise TypeError('%s cannot be used as argument type' % self)
        return ret

    def llvm_as_return(self):
        if not hasattr(self.desc, 'llvm_as_return'):
            ret = lc.Type.pointer(self.llvm_as_value())
        else:
            ret = self.desc.llvm_as_return()
        if ret is None:
            raise TypeError('%s cannot be used as return type' % self)
        return ret

    def llvm_value_from_arg(self, builder, arg):
        if not hasattr(self.desc, 'llvm_value_from_arg'):
            ret = arg
        else:
            ret = self.desc.llvm_value_from_arg(builder, arg)
        return ret

    def llvm_cast(self, builder, val, dst):
        if not hasattr(self.desc, 'llvm_cast'):
            raise TypeError('%s does not support casting '
                            'when trying to cast to %s' % (self, dst))
        ret = self.desc.llvm_cast(builder, val, dst.desc)
        if ret is None:
            raise TypeError('%s cannot be casted to %s' % (self, dst))
        return ret

    def llvm_cast_guarded(self, builder, raises, val, dst):
        '''optional. guarded cast that can raise exception
        '''
        if not hasattr(self.desc, 'llvm_cast_guarded'):
            ret = self.llvm_cast(builder, val, dst)
        else:
            ret = self.desc.llvm_cast_guarded(builder, raises, val, dst.desc)
        if ret is None:
            raise TypeError('%s cannot be casted to %s' % (self, dst))
        return ret

    def llvm_const(self, value):
        if not hasattr(self.desc, 'llvm_const'):
            raise TypeError('%s does not support constant value' % self)
        else:
            return self.desc.llvm_const(value)

    def llvm_unpack(self, builder, value):
        if not hasattr(self.desc, 'llvm_unpack'):
            raise TypeError('%s does not support unpacking' % self)
        else:
            return self.desc.llvm_unpack(builder, value)

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
        if isinstance(other, Type) and type(self.desc) == type(other.desc):
            return all(a == b
                        for a, b
                        in zip(self.list_fields(), Type(other).list_fields()))

    def __ne__(self, other):
        return not self.__eq__(other)

class Kind(object):
    def __init__(self, desc):
        self.desc = desc

    def __hash__(self):
        return hash((type(self), self.desc))

    def __eq__(self, other):
        if isinstance(other, Kind):
            return self.desc == other.desc

    def __ne__(self, other):
        return not self.__eq__(other)

    def matches(self, other):
        if isinstance(other, Type):
            return self.desc is type(other.desc)

    def __repr__(self):
        return '<kind %s>' % self.desc.__name__

class Boolean(object):
    fields = ()

    def __repr__(self):
        return 'bool'

    def coerce(self, other):
        if isinstance(other, Integer):
            return other.bitwidth // 8
        elif isinstance(other, Float):
            return other.bitwidth // 8 + 50
        elif isinstance(other, Complex):
            return other.bitwidth // 8 + 100
        elif isinstance(other, Boolean):
            return 0

    def llvm_as_value(self):
        return lc.Type.int(1)

    def llvm_const(self, value):
        return lc.Constant.int(self.llvm_as_value(), value)

    def llvm_cast(self, builder, value, dst):
        if isinstance(dst, Integer):
            return self.builder.zext(value, dst.llvm_as_value())
        elif isinstance(dst, Float):
            return self.builder.uitofp(value, dst.llvm_as_value())
        elif isinstance(dst, Complex):
            freal = self.llvm_cast(builder, value, dst.element.desc)
            fimag = dst.element.llvm_const(0)
            return dst.llvm_pack(builder, freal, fimag)

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
        elif isinstance(other, Boolean):
            return self.bitwidth

    def llvm_as_value(self):
        return lc.Type.int(self.bitwidth)

class Signed(Integer):
    def __init__(self, bitwidth):
        super(Signed, self).__init__(True, bitwidth)

    def llvm_const(self, value):
        return lc.Constant.int_signextend(self.llvm_as_value(), value)

    def llvm_cast(self, builder, value, dst):
        if isinstance(dst, Integer):
            if dst.bitwidth > self.bitwidth:
                return builder.sext(value, dst.llvm_as_value())
            else:
                return builder.trunc(value, dst.llvm_as_value())
        elif isinstance(dst, Float):
            return builder.sitofp(value, dst.llvm_as_value())
        elif isinstance(dst, Complex):
            freal = self.llvm_cast(builder, value, dst.element.desc)
            fimag = dst.element.llvm_const(0)
            return dst.llvm_pack(builder, freal, fimag)
        elif isinstance(dst, Boolean):
            zero = self.llvm_const(0)
            return builder.icmp(lc.ICMP_NE, value, zero)

    def llvm_cast_guarded(self, builder, raises, val, dst):
        if isinstance(dst, Integer):
            # check signed value to unsigned type
            if isinstance(dst, Unsigned):
                negative = signbit(builder, val)
                with cgutils.if_then(builder, negative):
                    raises(OverflowError, ERRMSG_NEGATIVE_TO_UNSIGNED)

            res = self.llvm_cast(builder, val, dst)

            # check change of sign
            if isinstance(dst, Signed):
                sbval = signbit(builder, val)
                sbres = signbit(builder, res)
                change_of_sign = builder.xor(sbres, sbval)
                with cgutils.if_then(builder, change_of_sign):
                    raises(OverflowError, ERRMSG_CHANGE_OF_SIGN)

            return res

        else:
            return self.llvm_cast(builder, val, dst)

    def ctype_as_argument(self):
        return getattr(ct, 'c_int%d' % self.bitwidth)

    def maximum_value(self):
        '''The constant maximum value
        '''
        all_significant = lc.Constant.all_ones(lc.Type.int(self.bitwidth - 1))
        return lc.Constant.zext(all_significant, lc.Type.int(self.bitwidth))

class Unsigned(Integer):
    def __init__(self, bitwidth):
        super(Unsigned, self).__init__(False, bitwidth)

    def llvm_const(self, value):
        return lc.Constant.int(self.llvm_as_value(), value)
    
    def llvm_cast(self, builder, value, dst):
        if isinstance(dst, Integer):
            if dst.bitwidth > self.bitwidth:
                return builder.zext(value, dst.llvm_as_value())
            else:
                return builder.trunc(value, dst.llvm_as_value())
        elif isinstance(dst, Float):
            return builder.uitofp(value, dst.llvm_as_value())
        elif isinstance(dst, Complex):
            freal = self.llvm_cast(builder, value, dst.element.desc)
            fimag = dst.element.llvm_const(0)
            return dst.llvm_pack(builder, freal, fimag)
        elif isinstance(dst, Boolean):
            zero = self.llvm_const(0)
            return builder.icmp(lc.ICMP_NE, value, zero)

    def llvm_cast_guarded(self, builder, raises, val, dst):
        if isinstance(dst, Integer):

            res = self.llvm_cast(builder, val, dst)
        
            # check change of sign
            if isinstance(dst, Signed):
                output_signed = signbit(builder, res)
                with cgutils.if_then(builder, output_signed):
                    raises(OverflowError, ERRMSG_CHANGE_OF_SIGN)

            return res
        else:
            return self.llvm_cast(builder, val, dst)

    def ctype_as_argument(self):
        return getattr(ct, 'c_uint%d' % self.bitwidth)

    def maximum_value(self):
        '''The constant maximum value
        '''
        return lc.Constant.all_ones(lc.Type.int(self.bitwidth))


class Float(object):
    fields = 'bitwidth',

    def __init__(self, bitwidth):
        self.bitwidth = bitwidth

    def coerce(self, other):
        if isinstance(other, Float):
            return (other.bitwidth - self.bitwidth) // 8
        elif isinstance(other, Integer):
            return (other.bitwidth - self.bitwidth) // 8 - 50 - 1
        elif isinstance(other, Complex):
            return (other.bitwidth - self.bitwidth) // 8 + 50
        elif isinstance(other, Boolean):
            return self.bitwidth + 50

    def __repr__(self):
        return 'float%d' % self.bitwidth

    def llvm_as_value(self):
        return {32: lc.Type.float(), 64: lc.Type.double()}[self.bitwidth]

    def llvm_const(self, value):
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
        elif isinstance(dst, Complex):
            elem = self.llvm_cast(builder, value, dst.element.desc)
            zero = dst.element.desc.llvm_const(0)
            return dst.llvm_pack(builder, elem, zero)
        elif isinstance(dst, Boolean):
            zero = lc.Constant.real(value.type, 0)
            return builder.fcmp(lc.FCMP_ONE, value, zero)

    def llvm_cast_guarded(self, builder, raises, val, dst):
        if isinstance(dst, Integer):
            nan = isnan(builder, val)
            with cgutils.if_then(builder, nan):
                raises(ValueError, ERRMSG_NAN)
        return self.llvm_cast(builder, val, dst)

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
        elif isinstance(other, Boolean):
            return self.bitwidth + 100

    def __repr__(self):
        return 'complex%d' % self.bitwidth

    def llvm_as_value(self):
        return lc.Type.struct([self.element.llvm_as_value()] * 2)

    def llvm_as_argument(self):
        return lc.Type.pointer(self.llvm_as_value())

    def llvm_const(self, value):
        real = self.element.llvm_const(value.real)
        imag = self.element.llvm_const(value.imag)
        return lc.Constant.struct([real, imag])

    def llvm_cast(self, builder, value, dst):
        if isinstance(dst, Complex):
            real, imag = self.llvm_unpack(builder, value)
            newelem = dst.element
            newreal = self.element.llvm_cast(builder, real, newelem)
            newimag = self.element.llvm_cast(builder, imag, newelem)
            return dst.llvm_pack(builder, newreal, newimag)

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
    fields = 'element', 'ndim', 'order'
    
    def __init__(self, element, ndim, order):
        assert order in ['C', 'CS', 'F', 'FS', 'A']
        self.ndim = ndim
        self.element = element
        self.order = order

    def coerce(self, other):
        return (self == other)

    def __repr__(self):
        return 'array(%s, %s, %s)' % (self.element, self.ndim, self.order)

    def llvm_as_value(self):
        lelem = lc.Type.pointer(self.element.llvm_as_value())
        lintp = intp.llvm_as_value()
        shapestrides = lc.Type.array(lintp, self.ndim)
        return lc.Type.struct([lelem, shapestrides, shapestrides])

    def llvm_as_argument(self):
        return lc.Type.pointer(self.llvm_as_value())

    def llvm_value_from_arg(self, builder, arg):
        return builder.load(arg)

    def llvm_as_return(self):
        '''does not support returning an array
        '''
        return

    def llvm_pack(self, builder, values):
        data, shape, strides = values
        ary = lc.Constant.undef(self.llvm_as_value())
        ary = builder.insert_value(ary, data, 0)
        ary = builder.insert_value(ary, shape, 1)
        ary = builder.insert_value(ary, strides, 2)
        return ary

    def ctype_as_argument(self):
        return ct.c_void_p

    def ctype_pack_argument(self, ary):
        c_array = make_array_ctype(ary.ndim)

        ary = c_array(data=ary.ctypes.data,
                      shape=ary.ctypes.shape,
                      strides=ary.ctypes.strides)
        return ct.byref(ary)

    def copy(self, ndim=None, order=None):
        ndim = ndim if ndim is not None else self.ndim
        order = order if order is not None else self.order
        return arraytype(element=self.element, ndim=ndim, order=order)

def make_array_ctype(ndim):
    c_intp = intp.ctype_as_argument()
    class c_array(ct.Structure):
            _fields_ = [('data',    ct.c_void_p),
                        ('shape',   c_intp * ndim),
                        ('strides', c_intp * ndim)]
    return c_array

ArrayKind = Kind(Array)

class Tuple(object):
    fields = 'elements',

    def __init__(self, elements):
        self.elements = elements

    def coerce(self, other):
        if isinstance(other, Tuple):
            return 0

    def __repr__(self):
        return 'tuple(%s)' % (', '.join(repr(e) for e in self.elements))

    def llvm_as_value(self):
        return lc.Type.struct([t.llvm_as_value() for t in self.elements])

    def llvm_as_argument(self):
        return

    def llvm_as_return(self):
        return
    
    def llvm_const(self, value):
        elems = [t.llvm_const(v) for t, v in zip(self.elements, value)]
        return lc.Constant.struct(elems)

    def llvm_pack(self, builder, items):
        out = lc.Constant.undef(self.llvm_as_value())
        for i, item in enumerate(items):
            out = builder.insert_value(out, item, i)
        return out

    def llvm_unpack(self, builder, tupleval):
        out = [builder.extract_value(tupleval, i)
               for i in range(len(self.elements))]
        return out

    def llvm_getitem(self, builder, tupleobj, index):
        return builder.extract_value(tupleobj, index)

TupleKind = Kind(Tuple)

class FixedArray(object):
    fields = 'element', 'length'

    def __init__(self, element, length):
        assert length > 0
        self.element = element
        self.length = length

    def coerce(self, other):
        if isinstance(other, FixedArray) and self.length == other.length:
            return self.element.try_coerce(other.element)

    def unpack(self, count):
        return (self.element,) * count

    def __repr__(self):
        return '[%s x %d]' % (self.element, self.length)

    def llvm_as_value(self):
        return lc.Type.array(self.element.llvm_as_value(), self.length)

    def llvm_as_argument(self):
        return

    def llvm_as_return(self):
        return
    
    def llvm_const(self, value):
        elems = [self.element.llvm_const(i) for i in value]
        return lc.Constant.array(self.element.llvm_as_value(), elems)

    def llvm_getitem(self, builder, aryobj, index):
        return builder.extract_value(aryobj, index)

    def llvm_pack(self, builder, values):
        out = lc.Constant.undef(self.llvm_as_value())
        for i, val in enumerate(values):
            assert val.type == out.type.element
            out = builder.insert_value(out, val, i)
        return out

    def llvm_unpack(self, builder, value):
        return [builder.extract_value(value, i) for i in range(self.length)]

FixedArrayKind = Kind(FixedArray)

class Slice2(object):
    '''Slice type that contains start and stop only
    '''
    fields = ()

    def __repr__(self):
        return 'slice2'

    def coerce(self, other):
        if isinstance(other, Slice2):
            return 0

    def llvm_as_value(self):
        llintp = intp.llvm_as_value()
        return lc.Type.struct([llintp] * 2)

    def llvm_const(self, value):
        start, stop = [intp.llvm_const(x)
                       for x in (value.start, value.stop)]
        return lc.Constant.struct([start, stop])

    def llvm_pack(self, builder, values):
        llintp = intp.llvm_as_value()
        out = lc.Constant.undef(self.llvm_as_value())
        if len(values) != 2:
            raise ValueError
        for i, val in enumerate(values):
            assert val.type == llintp
            out = builder.insert_value(out, val, i)
        return out

    def llvm_unpack(self, builder, val):
        return [builder.extract_value(val, i) for i in range(2)]

class Slice3(object):
    '''Slice type that contains start, stop, step
    '''
    fields = ()

    def __repr__(self):
        return 'slice3'

    def coerce(self, other):
        if isinstance(other, Slice3):
            return 0

    def llvm_as_value(self):
        llintp = intp.llvm_as_value()
        return lc.Type.struct([llintp] * 3)

    def llvm_const(self, value):
        start, stop, step = [intp.llvm_const(x)
                             for x in (value.start, value.stop, value.step)]
        return lc.Constant.struct([start, stop, step])

    def llvm_pack(self, builder, values):
        llintp = intp.llvm_as_value()
        out = lc.Constant.undef(self.llvm_as_value())
        if len(values) != 3:
            raise ValueError
        for i, val in enumerate(values):
            assert val.type == llintp
            out = builder.insert_value(out, val, i)
        return out

slice2 = Type(Slice2())
slice3 = Type(Slice3())

class BuiltinObject(object):
    fields = 'name',
    
    def __init__(self, name):
        self.name = name

    def coerce(self, other):
        if isinstance(other, BuiltinObject):
            return self.name == other.name

    def __repr__(self):
        return '<builtin %s>' % self.name

    def llvm_as_value(self):
        return lc.Type.pointer(lc.Type.int(8))


module_type = Type(BuiltinObject('module'))
function_type = Type(BuiltinObject('function'))
macro_type = Type(BuiltinObject('macro'))
method_type = Type(BuiltinObject('method'))
none_type = Type(BuiltinObject('none'))
exception_type = Type(BuiltinObject('exception'))

class IteratorFactory(object):
    fields = 'imp',

    def __init__(self, imp):
        self.imp = imp

    def coerce(self, other):
        if isinstance(other, IteratorFactory) and self.imp == other.imp:
            return 0

    def llvm_as_value(self):
        return self.imp.llvm_as_value()

    def __repr__(self):
        return repr(self.imp)

    def iterator(self):
        return self.imp.iterator()

IteratorFactoryKind = Kind(IteratorFactory)

class RangeType(BuiltinObject):
    def llvm_as_value(self):
        elem = intp.llvm_as_value()
        return lc.Type.struct((elem, elem, elem))

    def iterator(self):
        return range_iter_type

range_type = Type(IteratorFactory(RangeType('range')))

class RangeIterType(BuiltinObject):
    def llvm_as_value(self):
        return lc.Type.pointer(range_type.llvm_as_value())

    def iterate_data(self):
        return intp

class Iterator(object):
    fields = 'imp',

    def __init__(self, imp):
        self.imp = imp

    def coerce(self, other):
        if isinstance(other, Iterator) and self.imp == other.imp:
            return 0

    def llvm_as_value(self):
        return self.imp.llvm_as_value()

    def __repr__(self):
        return '<iter %s>' % self.imp

    def iterate_data(self):
        return self.imp.iterate_data()

IteratorKind = Kind(Iterator)

range_iter_type = Type(Iterator(RangeIterType('range-iter')))

class EnumerateType(object):
    fields = 'inner',

    def __init__(self, inner):
        self.inner = inner

    def coerce(self, other):
        if isinstance(other, EnumerateType) and self.inner == other.inner:
            return 0

    def llvm_as_value(self):
        counter = intp.llvm_as_value()
        inner = self.inner.llvm_as_value()
        return lc.Type.struct([counter, inner])

    def __repr__(self):
        return '<enumerate %s>' % self.inner

EnumerateKind = Kind(EnumerateType)

class EnumerateIterType(object):
    fields = 'state',
    
    def __init__(self, state):
        self.state = state

    def coerce(self, other):
        if isinstance(other, EnumerateIterType) and self.inner == other.inner:
            return 0

    def llvm_as_value(self):
        return lc.Type.pointer(self.state.llvm_as_value())

    def __repr__(self):
        return '<iter %s>' % self.state

    def inner(self):
        return self.state.desc.inner

    


EnumerateIterKind = Kind(EnumerateIterType)


void = Type(BuiltinObject('void'))

boolean = Type(Boolean())

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

def arraytype(element, ndim, order):
    return Type(Array(element, ndim, order))

def tupletype(*elements):
    elements = map(Type, elements)
    if all(elements[0] == other for other in elements[1:]):
        return Type(FixedArray(elements[0], len(elements)))
    else:
        return Type(Tuple(elements))

def enumerate_type(inner):
    return Type(EnumerateType(inner))

def enumerate_iter_type(state):
    return Type(EnumerateIterType(state))

intp = {4: int32, 8: int64}[tuple.__itemsize__]



#------------------------------------------------------------------------------
# utils

def signbit(builder, intval):
    shamt = lc.Constant.int(intval.type, intval.type.width - 1)
    shifted = builder.lshr(intval, shamt)
    return builder.trunc(shifted, lc.Type.int(1))

def isnan(builder, fltval):
    return builder.fcmp(lc.FCMP_UNO, fltval, fltval)

def const_intp(x):
    return intp.llvm_const(x)

def auto_intp(x):
    if isinstance(x, int):
        return const_intp(x)
    else:
        return x
