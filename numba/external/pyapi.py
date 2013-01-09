from external import ExternalFunction
from numba import *

class ofunc(ExternalFunction):
    arg_types = [object_]
    return_type = object_

class Py_IncRef(ofunc):
    # TODO: rewrite calls to Py_IncRef/Py_DecRef to direct integer
    # TODO: increments/decrements
    return_type = void

class Py_DecRef(Py_IncRef):
    pass

class PyObject_Length(ofunc):
    return_type = Py_ssize_t

class PyObject_Call(ExternalFunction):
    arg_types = [object_, object_, object_]
    return_type = object_

class PyObject_CallMethod(ExternalFunction):
    arg_types = [object_, c_string_type, c_string_type]
    return_type = object_
    is_vararg = True

class PyObject_Type(ExternalFunction):
    '''
        Added to aid debugging
        '''
    arg_types = [object_]
    return_type = object_

class PyTuple_Pack(ExternalFunction):
    arg_types = [Py_ssize_t]
    return_type = object_
    is_vararg = True

class Py_BuildValue(ExternalFunction):
    arg_types = [c_string_type]
    return_type = object_
    is_vararg = True

class PyArg_ParseTuple(ExternalFunction):
    arg_types = [object_, c_string_type]
    return_type = int_
    is_vararg = True

class PyObject_Print(ExternalFunction):
    arg_types = [object_, void.pointer(), int_]
    return_type = int_

class PyObject_Str(ExternalFunction):
    arg_types = [object_]
    return_type = object_

class PyObject_GetAttrString(ExternalFunction):
    arg_types = [object_, c_string_type]
    return_type = object_

class PyObject_GetItem(ExternalFunction):
    arg_types = [object_, object_]
    return_type = object_

class PyObject_SetItem(ExternalFunction):
    arg_types = [object_, object_, object_]
    return_type = int_

class PySlice_New(ExternalFunction):
    arg_types = [object_, object_, object_]
    return_type = object_

class PyErr_SetString(ExternalFunction):
    arg_types = [object_, c_string_type]
    return_type = void

class PyErr_Format(ExternalFunction):
    arg_types = [object_, c_string_type]
    return_type = void.pointer() # object_
    is_vararg = True

class PyErr_Occurred(ExternalFunction):
    arg_types = []
    return_type = void.pointer() # object_

class PyErr_Clear(ExternalFunction):
    arg_types = []
    return_type = void
#
### Object conversions to native types
#
def create_func(name, restype, argtype, d):
    class PyLong_FromLong(ExternalFunction):
        arg_types = [argtype]
        return_type = restype

    PyLong_FromLong.__name__ = name
    if restype.is_object:
        type = argtype
    else:
        type = restype

    d[type] = PyLong_FromLong
    globals()[name] = PyLong_FromLong

# The pipeline is using this dictionary to lookup casting func
_as_long = {}
def as_long(name, type):
    create_func(name, type, object_, _as_long)

as_long('PyLong_AsLong', long_)
as_long('PyLong_AsUnsignedLong', ulong)
as_long('PyLong_AsLongLong', longlong)
as_long('PyLong_AsUnsignedLongLong', ulonglong)
#as_long('PyLong_AsSize_t', size_t) # new in py3k
as_long('PyLong_AsSsize_t', Py_ssize_t)

class PyFloat_FromDouble(ExternalFunction):
    arg_types = [double]
    return_type = object_

class PyComplex_RealAsDouble(ExternalFunction):
    arg_types = [object_]
    return_type = double

class PyComplex_ImagAsDouble(ExternalFunction):
    arg_types = [object_]
    return_type = double

class PyComplex_FromDoubles(ExternalFunction):
    arg_types = [double, double]
    return_type = object_

class PyComplex_FromCComplex(ExternalFunction):
    arg_types = [complex128]
    return_type = object_

class PyInt_FromString(ExternalFunction):
    arg_types = [c_string_type, c_string_type.pointer(), int_]
    return_type = object_

class PyFloat_FromString(ExternalFunction):
    arg_types = [object_, c_string_type.pointer()]
    return_type = object_

class PyBool_FromLong(ExternalFunction):
    arg_types = [long_]
    return_type = object_

#
### Conversion of native types to object
#
# The pipeline is using this dictionary to lookup casting func
_from_long = {}

def from_long(name, type):
    create_func(name, object_, type, _from_long)

from_long('PyLong_FromLong', long_)
from_long('PyLong_FromUnsignedLong', ulong)
from_long('PyLong_FromLongLong', longlong)
from_long('PyLong_FromUnsignedLongLong', ulonglong)
from_long('PyLong_FromSize_t', size_t) # new in 2.6
from_long('PyLong_FromSsize_t', Py_ssize_t)

class PyFloat_AsDouble(ExternalFunction):
    arg_types = [object_]
    return_type = double

class PyComplex_AsCComplex(ExternalFunction):
    arg_types = [object_]
    return_type = complex128

def create_binary_pyfunc(name):
    class PyNumber_BinOp(ExternalFunction):
        arg_types = [object_, object_]
        return_type = object_
    PyNumber_BinOp.__name__ = name
    globals()[name] = PyNumber_BinOp

create_binary_pyfunc('PyNumber_Add')
create_binary_pyfunc('PyNumber_Subtract')
create_binary_pyfunc('PyNumber_Multiply')
create_binary_pyfunc('PyNumber_Divide')
create_binary_pyfunc('PyNumber_Remainder')

class PyNumber_Power(ExternalFunction):
    arg_types = [object_, object_, object_]
    return_type = object_

create_binary_pyfunc('PyNumber_Lshift')
create_binary_pyfunc('PyNumber_Rshift')
create_binary_pyfunc('PyNumber_Or')
create_binary_pyfunc('PyNumber_Xor')
create_binary_pyfunc('PyNumber_And')
create_binary_pyfunc('PyNumber_FloorDivide')

class PyNumber_Positive(ofunc):
    pass

class PyNumber_Negative(ofunc):
    pass

class PyNumber_Invert(ofunc):
    pass

class PyObject_IsTrue(ExternalFunction):
    arg_types = [object_]
    return_type = int_

__all__ = [k for k, v in globals().items()
           if (v != ExternalFunction
               and isinstance(v, type)
               and issubclass(v, ExternalFunction))]
