# Copyright (c) 2012, Siu Kwan Lam
# All rights reserved.

import ctypes

from pymothoa.util.descriptor import Descriptor, instanceof
from pymothoa import types

import llvm # binding
import values

_array_type_code_to_ctype = {
    'c': ctypes.c_char,
    'b': ctypes.c_ubyte,
    'B': ctypes.c_byte,
    'h': ctypes.c_short,
    'H': ctypes.c_ushort,
    'i': ctypes.c_int,
    'I': ctypes.c_uint,
    'l': ctypes.c_long,
    'L': ctypes.c_ulong,
    'f': ctypes.c_float,
    'd': ctypes.c_double,
}

class LLVMType(object):

    def __new__(cls, datatype):
        TYPE_MAP = {
            types.Void      : LLVMVoid,
            types.Bool      : LLVMBool,
            types.Int8      : LLVMInt8,
            types.Int16     : LLVMInt16,
            types.Int32     : LLVMInt32,
            types.Int64     : LLVMInt64,
            types.Float     : LLVMFloat,
            types.Double    : LLVMDouble,
        }
        try:
            return object.__new__(TYPE_MAP[datatype])
        except KeyError:
            def is_array(datatype):
                if isinstance(datatype, types.Array):
                    return True
                elif type(datatype) is not type:
                    return False
                elif issubclass(datatype, types.GenericBoundedArray):
                    return True
                elif issubclass(datatype, types.GenericUnboundedArray):
                    return True
                else:
                    return False

            if is_array(datatype): # array type
                elemtype = datatype.elemtype
                obj = object.__new__(LLVMUnboundedArray)
                obj.elemtype = LLVMType(elemtype)
                return obj
            elif type(datatype) is type and issubclass(datatype, types.GenericVector):
                elemtype = LLVMType(datatype.elemtype)
                elemcount = datatype.elemcount
                # determine mixin classes to install
                if isinstance(elemtype, LLVMRealBinOpMixin):
                    mixins = (LLVMRealBinOpMixin ,)
                elif isinstance(elemtype, LLVMIntBinOpMixin):
                    mixins = (LLVMIntBinOpMixin ,)
                else:
                    raise NotImplementedError
                # create new class for the vector type
                clsname = 'LLVMVector__%s%d'%(datatype.elemtype.__name__, elemcount)
                vectorcls = type(clsname, (LLVMVector,)+mixins, {})
                # create instance of the class
                obj = object.__new__(vectorcls)
                obj.elemtype = elemtype
                obj.elemcount = elemcount
                return obj
            else:
                raise TypeError(datatype)

class LLVMVoid(types.Void):
    def ctype(self):
        return None

    def type(self):
        return llvm.TypeFactory.make_void()

class LLVMBool(types.Bool):
    def ctype(self):
        return NotImplementedError('No equivalence ctype for 1-bit boolean')

    def type(self):
        return llvm.TypeFactory.make_int(1) # 1-bit integer

    def cast(self, old, builder):
        if old.type == self:
            return old.value(builder)
        elif isinstance(old.type, types.GenericInt) or isinstance(old.type, types.GenericReal):
            return old.type.op_eq(old.value(builder), old.type.constant(0), builder)
        else:
            raise TypeError('Cannot cast to boolean.'+'%s %s'%(old.type, self))

    def op_not(self, value, builder):
        return builder.bitwise_neg(value)

class LLVMIntBinOpMixin(object):
    def op_add(self, lhs, rhs, builder):
        return builder.add(lhs, rhs)

    def op_sub(self, lhs, rhs, builder):
        return builder.sub(lhs, rhs)

    def op_mult(self, lhs, rhs, builder):
        return builder.mul(lhs, rhs)

    def op_div(self, lhs, rhs, builder):
        return builder.sdiv(lhs, rhs)

    def op_mod(self, lhs, rhs, builder):
        return builder.smod(lhs, rhs)

    def op_eq(self, lhs, rhs, builder):
        return builder.icmp(llvm.ICMP_EQ, lhs, rhs)

    def op_noteq(self, lhs, rhs, builder):
        return builder.icmp(llvm.ICMP_NE, lhs, rhs)

    def op_lt(self, lhs, rhs, builder):
        return builder.icmp(llvm.ICMP_SLT, lhs, rhs)

    def op_lte(self, lhs, rhs, builder):
        return builder.icmp(llvm.ICMP_SLE, lhs, rhs)

    def op_gt(self, lhs, rhs, builder):
        return builder.icmp(llvm.ICMP_SGT, lhs, rhs)

    def op_gte(self, lhs, rhs, builder):
        return builder.icmp(llvm.ICMP_SGE, lhs, rhs)

    def op_bitand(self, lhs, rhs, builder):
        return builder.bitwise_and(lhs, rhs)

    def op_bitor(self, lhs, rhs, builder):
        return builder.bitwise_or(lhs, rhs)

class LLVMBasicIntMixin(LLVMIntBinOpMixin):

    def argument_adaptor(self, val):
        return val

    def ctype(self):
        mapping = {
             8: ctypes.c_int8,
            16: ctypes.c_int16,
            32: ctypes.c_int32,
            64: ctypes.c_int64,
        }
        return mapping[self.bitsize]

    def type(self):
        return llvm.TypeFactory.make_int(self.bitsize)

    def constant(self, val):
        if type(val) is not int:
            raise TypeError(type(val))
        else:
            return llvm.ConstantFactory.make_int(self.type(), val)

    def cast(self, old, builder):
        if old.type == self:
            return old.value(builder)
        elif isinstance(old.type, types.Bool):
            return builder.icast(old.value(builder), self.type(), True)
        elif isinstance(old.type, types.GenericInt):
            assert old.type.bitsize != self.bitsize
            val = old.value(builder)
            return builder.icast(val, self.type(), self.signed)
        elif isinstance(old.type, types.GenericReal):
            val = old.value(builder)
            return builder.fptosi(val, self.type())
        else:
            print 'cast %s -> %s'%(old.type, self)

        assert False


class LLVMRealBinOpMixin(object):
    def op_add(self, lhs, rhs, builder):
        return builder.fadd(lhs, rhs)

    def op_sub(self, lhs, rhs, builder):
        return builder.fsub(lhs, rhs)

    def op_mult(self, lhs, rhs, builder):
        return builder.fmul(lhs, rhs)

    def op_div(self, lhs, rhs, builder):
        return builder.fdiv(lhs, rhs)

    def op_mod(self, lhs, rhs, builder):
        return builder.fmod(lhs, rhs)

    def op_eq(self, lhs, rhs, builder):
        return builder.fcmp(llvm.FCMP_OEQ, lhs, rhs)

    def op_lt(self, lhs, rhs, builder):
        return builder.fcmp(llvm.FCMP_OLT, lhs, rhs)

    def op_lte(self, lhs, rhs, builder):
        return builder.fcmp(llvm.FCMP_OLE, lhs, rhs)

    def op_gt(self, lhs, rhs, builder):
        return builder.fcmp(llvm.FCMP_OGT, lhs, rhs)

    def op_gte(self, lhs, rhs, builder):
        return builder.fcmp(llvm.FCMP_OGE, lhs, rhs)

class LLVMBasicFloatMixin(LLVMRealBinOpMixin):

    def argument_adaptor(self, val):
        return val

    def ctype(self):
        return ctypes.c_float

    def type(self):
        return llvm.TypeFactory.make_float()

    def constant(self, val):
        return llvm.ConstantFactory.make_real(self.type(), val)

    def cast(self, old, builder):
        if old.type == self:
            return old.value(builder)
        elif isinstance(old.type, types.GenericInt):
            val = old.value(builder)
            return builder.sitofp(val, self.type())
        elif isinstance(old.type, types.GenericReal):
            val = old.value(builder)
            return builder.fcast(val, self.type())
        else:
            print 'cast %s -> %s'%(old.type, self)

        assert False

class LLVMBasicDoubleMixin(LLVMBasicFloatMixin):
    def ctype(self):
        return ctypes.c_double

    def type(self):
        return llvm.TypeFactory.make_double()

class LLVMInt8(types.Int8, LLVMBasicIntMixin):
    pass

class LLVMInt16(types.Int16, LLVMBasicIntMixin):
    pass

class LLVMInt32(types.Int32, LLVMBasicIntMixin):
    pass

class LLVMInt64(types.Int64, LLVMBasicIntMixin):
    pass

class LLVMFloat(types.Float, LLVMBasicFloatMixin):
    pass

class LLVMDouble(types.Float, LLVMBasicDoubleMixin):
    pass

class LLVMUnboundedArray(types.GenericUnboundedArray):

    elemtype = Descriptor(constant=True, constrains=instanceof(types.BuiltinType))

    def cast(self, old, builder):
        if isinstance(old.type, LLVMUnboundedArray):
            return old.value(builder)
        else:
            raise TypeError('Casting unbounded-array to something else.')

    def ctype(self):
        return ctypes.POINTER(self.elemtype.ctype())

    def type(self):
        return llvm.TypeFactory.make_pointer(self.elemtype.type())

    def argument_adaptor(self, val):
        try: # try to use numpy.ndarray
            from numpy import ndarray
            if isinstance(val, ndarray):
                if val.dtype != self.elemtype.ctype():
                    raise TypeError('dtype of the numpy.ndarray '
                                    'does not match argument type.')
                return val.ctypes.data_as(self.ctype())
        except ImportError:
            pass

        # No numpy or val is not ndarray.
        # Try to use array.array
        from array import array
        if isinstance(val, array):
            elemctype = self.elemtype.ctype()

            if _array_type_code_to_ctype[val.typecode]!=elemctype:
                raise TypeError('array.array contains a different datatype.')

            address, length = val.buffer_info()
            ptr = ctypes.cast(address, self.ctype())
            return ptr

        # Build a ctype array from iterable. This can be very slow.
        ctype = self.elemtype.ctype()
        argtype = ctype*len(val)
        return argtype(*val)

class LLVMVector(types.GenericVector):
    elemtype = Descriptor(constant=True, constrains=instanceof(types.BuiltinType))
    elemcount = Descriptor(constant=True, constrains=lambda N: N>1)

    def type(self):
        return llvm.TypeFactory.make_vector(self.elemtype.type(), self.elemcount)

    def cast(self, old, builder):
        from values import LLVMTempValue, LLVMConstant
        if not isinstance(old.type, LLVMVector): # from scalar to vector
            elem_casted = self.elemtype.cast(old, builder)
            vector = llvm.ConstantFactory.make_undef(self.type())
            for i in range(self.elemcount):
                idx = LLVMConstant(LLVMType(types.Int), i).value(builder)
                vector = builder.insert_element(vector, elem_casted, idx)
            return vector
        if old.type.elemtype != self.elemtype:
            raise TypeError('Invalid casting')

        return old.value(builder)

    def coerce(self, other):
        if not isinstance(other, LLVMVector): # other is of scalar type
            # then, try to promote it to a vector type
            return self

        if other.elemtype != self.elemtype:
            raise TypeError('Invalid casting')

        return self

    def argument_adaptor(self, val):
        raise NotImplementedError('Cannot use vector as argument.')
