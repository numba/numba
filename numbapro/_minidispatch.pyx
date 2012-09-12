from utils cimport *

cdef class MiniFunction(Function):
    cdef minifunc *func
    def __init__(self, Py_intptr_t func):
        self.func = <minifunc *> func

    cdef int invoke(self, cnp.npy_intp *shape, char **data_pointers,
                    cnp.npy_intp **strides_pointers) nogil except -1:
        return self.func(shape, data_pointers, strides_pointers)

cdef class MiniUFuncDispatcher(UFuncDispatcher):
    """
    Given a dict of signatures mapping to functions, dispatch to the right
    ufunc for element-wise operations.

    functions: a dict mapping (dtypeA, dtypeB, dimensionality) ->
                                    (func_pointer, ctypes_func, ctypes_ret_type,
                                     ctypes_arg_type, result_dtype)
    """

    def __init__(self, functions, nin, parallel):
        super(MiniUFuncDispatcher, self).__init__(functions, nin, parallel)
        for key, functions in functions.items():
            functions[key] = tuple(MiniFunction(function)
                                       for function in functions)

    def key(self, arg_dtypes, broadcast):
        return arg_dtypes + (min(broadcast.nd, 2),)
