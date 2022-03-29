from itertools import product

from numba.core import types
from ._vector_type_meta import (
    _vector_type_prefix,
    _vector_type_attribute_names,
    _vector_type_element_counts
)


class Dim3(types.Type):
    """
    A 3-tuple (x, y, z) representing the position of a block or thread.
    """
    def __init__(self):
        super().__init__(name='Dim3')


class GridGroup(types.Type):
    """
    The grid of all threads in a cooperative kernel launch.
    """
    def __init__(self):
        super().__init__(name='GridGroup')


dim3 = Dim3()
grid_group = GridGroup()


class CUDADispatcher(types.Dispatcher):
    """The type of CUDA dispatchers"""
    # This type exists (instead of using types.Dispatcher as the type of CUDA
    # dispatchers) so that we can have an alternative lowering for them to the
    # lowering of CPU dispatchers - the CPU target lowers all dispatchers as a
    # constant address, but we need to lower to a dummy value because it's not
    # generally valid to use the address of CUDA kernels and functions.
    #
    # Notes: it may be a bug in the CPU target that it lowers all dispatchers to
    # a constant address - it should perhaps only lower dispatchers acting as
    # first-class functions to a constant address. Even if that bug is fixed, it
    # is still probably a good idea to have a separate type for CUDA
    # dispatchers, and this type might get other differentiation from the CPU
    # dispatcher type in future.


# Support for CUDA built-in Vector Types
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types


class VectorType(types.Type):
    """Meta class for all vector types."""
    def __init__(self, name, base_type, attr_names):
        self._base_type = base_type
        self._attr_names = attr_names
        super().__init__(name=name)

    @property
    def base_type(self):
        return self._base_type

    @property
    def attr_names(self):
        return self._attr_names

    @property
    def num_elements(self):
        return len(self._attr_names)


_vector_type_to_base_types = {
    "char": types.char,
    "short": types.int16,
    "int": types.int32,
    "long": types.int32,
    "longlong": types.int64,
    "uchar": types.uchar,
    "ushort": types.uint16,
    "uint": types.uint32,
    "ulong": types.uint32,
    "ulonglong": types.uint64,
    "float": types.float32,
    "double": types.float64
}

vector_type_list = []


def make_vector_type(name, nelem, base_type):
    type_name = f"{name}{nelem}"
    attr_names = _vector_type_attribute_names[:nelem]
    vector_type = type(
        type_name, (VectorType,), {})(type_name, base_type, attr_names)
    return vector_type


for name, nelem in product(_vector_type_prefix, _vector_type_element_counts):
    base_type = _vector_type_to_base_types[name]
    vector_type_list.append(make_vector_type(name, nelem, base_type))

print("vector_type_list", vector_type_list)
