from numba.typesystem import *
from numba.minivect import minitypes

#------------------------------------------------------------------------
# Container Types
#------------------------------------------------------------------------

class ContainerListType(NumbaKeyHashingType, minitypes.ObjectType):

    is_container = True
    subtypes = ['base_type']

    def __init__(self, base_type, size=-1):
        super(ContainerListType, self).__init__()
        self.base_type = base_type
        self.size = size

    @property
    def key(self):
        return self.base_type, self.size

    def is_sized(self):
        return self.size >= 0

    def __repr__(self):
        return "%s(%s, %s)" % (self.name, self.base_type, self.size)

class TupleType(ContainerListType):

    is_tuple = True
    name = "tuple"

class ListType(ContainerListType):
    is_list = True
    name = "list"

class MapContainerType(NumbaType):

    is_map = True

    def __init__(self, key_type, value_type, size=-1):
        super(MapContainerType, self).__init__()
        self.key_type = key_type
        self.value_type = value_type
        self.size = size

class DictType(MapContainerType, minitypes.ObjectType):

    is_dict = True
    name = "dict"
    size = 0

    def __str__(self):
        return "dict(%s)" % ", ".join(["..."] * self.size)

#------------------------------------------------------------------------
# Natively Typed Container Types
#------------------------------------------------------------------------

class TypedContainerListType(ContainerListType):

    is_typed_container = True

class TypedMapContainerType(MapContainerType):

    is_typed_map = True

class TypedTupleType(TypedContainerListType):

    is_typed_tuple = True
    name = "typedtuple"

class TypedListType(TypedContainerListType):

    is_typed_list = True
    name = "typedlist"

class TypedSetType(TypedContainerListType):

    is_typed_set = True
    name = "typedset"

class TypedFrozenSetType(TypedContainerListType):

    is_typed_frozenset = True
    name = "typedfrozenset"

class TypedDictType(TypedMapContainerType, minitypes.ObjectType):

    is_typed_dict = True
    name = "typeddict"

#------------------------------------------------------------------------
# User-facing constructors
#------------------------------------------------------------------------

def typedlist(base_type, size=-1):
    return TypedTupleType(base_type, size)

def typedtuple(base_type, size=-1):
    return TypedListType(base_type, size)

def typedset(base_type, size=-1):
    return TypedSetType(base_type, size)

def typedfrozenset(base_type, size=-1):
    return TypedSetType(base_type, size)

def typeddict(base_type, size=-1):
    return TypedFrozenSetType(base_type, size)


#------------------------------------------------------------------------
# Shorthands
#------------------------------------------------------------------------

tuple_ = TupleType(object_)
list_ = ListType(object_)
dict_ = DictType(object_, object_)