cimport numpy as cnp

cdef extern from "Python.h":
    ctypedef int Py_intptr_t
    ctypedef unsigned int Py_uintptr_t

ctypedef int minifunc(cnp.npy_intp *shape, char **data_pointers,
                      cnp.npy_intp **strides_pointers) nogil except -1

cdef class Function(object):
    cdef int invoke(self, cnp.npy_intp *shape, char **data_pointers,
                    cnp.npy_intp **strides_pointers) nogil except -1

cdef class UFuncDispatcher(object):

    cdef object functions
    cdef int nin
    cdef bint parallel
    cdef int max_specialization_ndim

    cdef _flatten_contig_shape(
        self, cnp.npy_intp *shape_p, cnp.npy_intp **strides, int ndim,
        int n_ops, bint contig, order)

    cdef run_ufunc(self, Function function, broadcast, int ndim, out,
                   list arrays, bint contig, order)

    cdef int run_higher_dimensional(
            self, Function function, cnp.npy_intp *shape,
            char **data_pointers, cnp.npy_intp **strides,
            int ndim, int n_ops, int dim_level, bint contig) nogil except -1

    cdef int run_higher_dimensional_parallel(
            self, Function function, cnp.npy_intp *shape,
            char **data_pointers, cnp.npy_intp **strides,
            int ndim, int n_ops, int dim_level, bint contig) nogil except -1

    cdef int run_ufunc_parallel(
            self, Function function, cnp.npy_intp *shape,
            char **data_pointers, cnp.npy_intp **strides,
            int ndim, int n_ops, int dim_level, bint contig) except -1
