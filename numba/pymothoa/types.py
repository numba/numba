# Copyright (c) 2012, Siu Kwan Lam
# All rights reserved.

class Type(object):
    __init__ = NotImplemented

    def __eq__(self, other):
        return type(self) is type(other)

    def __ne__(self, other):
        return not (self==other)

class BuiltinType(Type):
    def coerce(self, other):
        '''Returns the type that has a higher rank.
        '''
        if self.rank > other.rank:
            return self
        else:
            return other

class Void(BuiltinType):
    rank = 0

class GenericInt(BuiltinType):
    pass

class Bool(BuiltinType):
    rank = 5

class Int8(GenericInt):
    rank = 10
    bitsize = 8
    signed = True


class Int16(GenericInt):
    rank = 11
    bitsize = 16
    signed = True


class Int32(GenericInt):
    rank = 12
    bitsize = 32
    signed = True

class Int64(GenericInt):
    rank = 13
    bitsize = 64
    signed = True

def _determine_native_int_size():
    from ctypes import c_int, c_int32, c_int64
    if c_int is c_int32:
        return Int32
    elif c_int is c_int64:
        return Int64
    else:
        raise NotImplementedError('Integer size other than 32/64-bit?')

Int = _determine_native_int_size()

class GenericReal(BuiltinType):
    pass

class Float(GenericReal):
    rank = 20

class Double(GenericReal):
    rank = 30

class AggregateType(Type):
    pass

class GenericUnboundedArray(AggregateType):
    pass

class GenericBoundedArray(AggregateType):
    pass

class GenericVector(Type):
    pass


class DummyType(Type):
    '''Dummy type class for allowing the backend to recognize special type constructions.
    '''
    pass

class Vector(DummyType):
    def __init__(self, ty, ct):
        pass

class Array(DummyType):
    __slots__ = 'elemtype'
    def __init__(self, elemtype, *ignored):
        '''Note: elemcount is ignored when using in the Python scope.
        '''
        self.elemtype = elemtype

class Slice(DummyType):
    def __init__(self, elemtype):
        pass
