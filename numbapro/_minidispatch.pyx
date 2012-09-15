from utils cimport *

cdef class MiniFunction(Function):

    cdef minifunc *func

    def __init__(self, Py_uintptr_t func):
        self.func = <minifunc *> func

    cdef int invoke(self, cnp.npy_intp *shape, char **data_pointers,
                    cnp.npy_intp **strides_pointers) nogil except -1:
        return self.func(shape, data_pointers, strides_pointers)

cdef class MiniUFuncDispatcher(UFuncDispatcher):

    def __init__(self, functions, nin, parallel):
        super(MiniUFuncDispatcher, self).__init__(functions, nin, parallel)
        for key, (kernels, result_dtype) in functions.items():
            kernels = tuple(MiniFunction(kernel) for kernel in kernels)
            functions[key] = (kernels, result_dtype)

    def key(self, arg_dtypes, broadcast, contig, inner_contig, tiled):
        if contig:
            nd = 1
        else:
            nd = broadcast.nd

        return arg_dtypes + (min(broadcast.nd, 2),)
