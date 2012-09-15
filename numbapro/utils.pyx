cimport cython
cimport cython.parallel
from libc cimport stdlib, string as libc_string
from cpython cimport *
cimport openmp

cimport numpy as cnp
cimport utils as cutils

import logging

from numba import visitors as numba_visitors
from numba.minivect import (miniast,
                            minitypes,
                            specializers as minispecializers,
                            ctypes_conversion)
from numba import decorators, utils, functions

from numbapro import _internal
from numbapro.vectorize import _common, basic

import numpy as np

include "miniutils.pyx"

cdef extern from "_internal.h":
    ctypedef struct PyDynUFuncObject:
        PyObject *minivect_dispatcher
        PyObject *cuda_dispatcher

def set_dispatchers(ufunc, mini_dispatcher, cuda_dispatcher):
    assert isinstance(ufunc, _internal.dyn_ufunc)

    if mini_dispatcher is not None:
        (<PyDynUFuncObject *> ufunc).minivect_dispatcher = <PyObject *> mini_dispatcher
        Py_INCREF(mini_dispatcher)

    if cuda_dispatcher is not None:
        (<PyDynUFuncObject *> ufunc).cuda_dispatcher = <PyObject *> cuda_dispatcher
        Py_INCREF(cuda_dispatcher)

cdef class Function(object):
    "Wraps a minivect or cuda function"

    cdef int invoke(self, cnp.npy_intp *shape, char **data_pointers,
                    cnp.npy_intp **strides_pointers) nogil except -1:
        with gil:
            raise NotImplementedError

cdef class UFuncDispatcher(object):
    """
    Given a dict of signatures mapping to functions, dispatch to the right
    ufunc for element-wise operations.

    functions:

        A dict mapping

            self.key((dtypeA, dtypeB), broadcast(*input_arrays))

        to

            (result_dtype, (contig_func, inner_contig_func,
                            tiled_func, strided_func))
    """

    def __init__(self, functions, nin, parallel, max_specialization_ndim=2):
        self.max_specialization_ndim = max_specialization_ndim
        self.functions = functions
        self.nin = nin
        self.parallel = parallel

    def key(self, arg_dtypes, broadcast, contig, inner_contig, tiled):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        """
        ufunc(array1, array2, ...)

        This has to figure out the data order, select the specialization,
        perform broadcasting of shape and strides, check for overlapping
        memory, and invoke the minivect kernel.
        """
        cdef int ndim

        if len(args) != self.nin:
            raise TypeError("Expected %d input arguments, got %d" %
                                                (self.nin, len(args)))

        args = list(map(np.asarray, args))
        arg_types = tuple(arg.dtype for arg in args)

        out = kwds.pop('out', None)
        order = _internal.get_arrays_ordering(args)

        broadcast_args = list(args)
        if out is not None:
            broadcast_args.append(out)

        # Broadcast arrays
        broadcast = np.broadcast(*broadcast_args)
        ndim = broadcast.nd

        # Figure out whether we have broadcasting operands, and the
        # overall data order
        any_broadcasting = is_broadcasting(broadcast_args, broadcast)
        contig = (order & _internal.ARRAYS_ARE_CONTIG) and not any_broadcasting
        inner_contig = order & _internal.ARRAYS_ARE_INNER_CONTIG
        tiled = order & (_internal.ARRAYS_ARE_MIXED_CONTIG|
                         _internal.ARRAYS_ARE_MIXED_STRIDED)

        if order & _internal.ARRAY_C_ORDER:
            order_arg = 'C'
        else:
            order_arg = 'F'

        # Find ufunc from input arrays
        key = self.key(arg_types, broadcast, contig, inner_contig, tiled)
        if key not in self.functions:
            raise ValueError(
                "No signature found for input types (%s, ndim=%d)" % (
                    ", ".join(str(dtype) for dtype in arg_types), broadcast.nd))

        functions, result_dtype = self.functions[key]

        # Instantiate the LHS
        if out is None:
            out = np.empty(broadcast.shape, dtype=result_dtype, order=order_arg)
        else:
            # TODO: check for read-after-write
            pass

        contig_func, inner_contig_func, tiled_func, strided_func = functions

        logging.info('contig: %s, inner_contig: %s, tiled: %s' %
                                        (contig, inner_contig, tiled))
        if contig:
            function = contig_func
        elif inner_contig:
            function = inner_contig_func
#        elif tiled:
#            function = tiled_func
        else:
            function = strided_func

        return self.run_ufunc(function, broadcast, ndim, out, args, contig,
                              order_arg)

    cdef _flatten_contig_shape(
            self, cnp.npy_intp *shape_p, cnp.npy_intp **strides, int ndim,
            int n_ops, bint contig, order):
        """
        Flatten shape for contiguous specializations, and set strides[0] to
        the smallest stride (for parallel invocation).

        This allows us to compile only a single, 1D, specialization.
        """
        cdef int i

        if ndim > 1:
            for i in range(1, ndim):
                shape_p[0] *= shape_p[i]

            print shape_p[0]
            if order == 'C':
                for i in range(n_ops):
                    strides[i][0] = strides[i][ndim - 1]

    cdef run_ufunc(self, Function function, broadcast, int ndim, out,
                   list arrays, bint contig, order):
        """
        Broadcast the shape, build the arguments and run the ufunc.
        """
        cdef cnp.npy_intp *shape_p, *strides_p
        cdef cnp.npy_intp **strides_args
        cdef char **data_pointers

        arrays.insert(0, out)
        broadcast_arrays(arrays, broadcast.shape, ndim, &shape_p, &strides_p)
        build_dynamic_args(arrays, strides_p, &data_pointers, &strides_args,
                           ndim)

        if contig:
            self._flatten_contig_shape(shape_p, strides_args, ndim,
                                       len(arrays), contig, order)
            ndim = 1

        try:
            if self.parallel and ndim > self.max_specialization_ndim:
                self.run_higher_dimensional_parallel(
                        function, shape_p, data_pointers, strides_args, ndim,
                        len(arrays), 0, contig)
            elif self.parallel:
                self.run_ufunc_parallel(
                    function, shape_p, data_pointers, strides_args, ndim,
                    len(arrays), 0, contig)
            else:
                self.run_higher_dimensional(
                        function, shape_p, data_pointers, strides_args, ndim,
                        len(arrays), 0, contig)
        finally:
            stdlib.free(shape_p)
            stdlib.free(strides_p)
            stdlib.free(data_pointers)
            stdlib.free(strides_args)

        return arrays[0]

    cdef int run_higher_dimensional(
            self, Function function, cnp.npy_intp *shape,
            char **data_pointers, cnp.npy_intp **strides,
            int ndim, int n_ops, int dim_level, bint contig) nogil except -1:
        """
        Run the 1 or 2 dimensional ufunc. If ndim > 2, we need a to simulate
        an outer loop nest of depth ndim - 2.
        """
        cdef int i, j

        if ndim <= self.max_specialization_ndim or contig:
            # Offset the stride pointer to remove the strides for
            # preceding dimensions
            # TODO: have minivect accept a dim_level argument
            if dim_level:
                for j in range(n_ops):
                    strides[j] += dim_level

            function.invoke(&shape[dim_level], data_pointers, strides)

            # Reset the stride pointers to the first stride for each operand
            if dim_level:
                for j in range(n_ops):
                    strides[j] -= dim_level
        else:
            # ndim > 2
            for i in range(shape[dim_level]):
                self.run_higher_dimensional(function, shape,
                                            data_pointers, strides, ndim - 1,
                                            n_ops, dim_level + 1, contig)

                # Add stride in this dimension to the data pointers of the
                # arrays
                for j in range(n_ops):
                    data_pointers[j] += strides[j][dim_level]

            # Back up the data pointers, so the outer dimension can add its
            # stride
            for j in range(n_ops):
                data_pointers[j] -= shape[dim_level] * strides[j][dim_level]

        return 0

    cdef int run_higher_dimensional_parallel(
            self, Function function, cnp.npy_intp *shape,
            char **data_pointers, cnp.npy_intp **strides,
            int ndim, int n_ops, int dim_level, bint contig) nogil except -1:
        """
        Run the ufunc using OpenMP on the outermost loop level.
        """
        cdef int i, j

        cdef char **data_pointers_copy
        cdef cnp.npy_intp **strides_pointers_copy

        with cython.parallel.parallel():
            data_pointers_copy = <char **> stdlib.malloc(
                                                    n_ops * sizeof(char **))
            strides_pointers_copy = <cnp.npy_intp **> stdlib.malloc(
                                            n_ops * sizeof(cnp.npy_intp **))

            try:
                if data_pointers_copy == NULL or strides_pointers_copy == NULL:
                    with gil:
                        raise MemoryError

                # libc_string.memcpy(data_pointers_copy, data_pointers,
                #                    n_ops * sizeof(char **))
                libc_string.memcpy(strides_pointers_copy, strides,
                                   n_ops * sizeof(cnp.npy_intp **))

                for i in cython.parallel.prange(shape[0]):
                    for j in range(n_ops):
                        # Add stride in this dimension to the data pointers of the
                        # arrays
                        data_pointers_copy[j] = (data_pointers[j] +
                                                 i * strides[j][dim_level])

                    self.run_higher_dimensional(
                            function, shape, data_pointers_copy,
                            strides_pointers_copy, ndim - 1, n_ops,
                            dim_level + 1, contig)
            finally:
                stdlib.free(data_pointers_copy)
                stdlib.free(strides_pointers_copy)

        return 0

    cdef int run_ufunc_parallel(
            self, Function function, cnp.npy_intp *shape,
            char **data_pointers, cnp.npy_intp **strides,
            int ndim, int n_ops, int dim_level, bint contig) except -1:
        """
        Run a minivect kernel in parallel:

            1) Each kernel gets shape (shape[0] / num_threads, shape[1], ...)
            2) The last kernel gets any remaining iterations that could not be
               divided
            3) Create a copy of the data pointers for each thread, and apply
               the offsets for the first dimension
        """
        cdef int i, j
        cdef int thread_num, num_threads
        cdef int N

        cdef char **data_pointers_copy
        cdef cnp.npy_intp last_shape[8]
        cdef cnp.npy_intp *shape_arg

        #cdef openmp.omp_lock_t lock
        #openmp.omp_init_lock(&lock)

        if ndim > 8:
            # If you get this exception, increase the size of 'last_shape'
            # This should not happen, since we only specialize for 1D and 2D
            raise Exception("max_specialization_ndim must be <= 8")

        # Populate shape for the last thread
        for i in range(ndim):
            last_shape[i] = shape[i]

        with nogil, cython.parallel.parallel():
            data_pointers_copy = <char **> stdlib.malloc(
                                                    n_ops * sizeof(char **))
            if data_pointers_copy == NULL:
                with gil:
                    raise MemoryError

            thread_num = openmp.omp_get_thread_num()
            num_threads = openmp.omp_get_num_threads()

            with cython.cdivision(True):
                N = shape[0] / num_threads

            # Apply offsets
            for i in range(n_ops):
                data_pointers_copy[i] = ((<char *> data_pointers[i]) +
                                             thread_num * N * strides[i][0])

            # Set the shape argument
            if thread_num == num_threads - 1:
                shape_arg = last_shape
            else:
                shape_arg = shape

            # Let a single thread divide the work
            for j in cython.parallel.prange(1):
                shape[0] = N
                with cython.cdivision(True):
                    last_shape[0] = N + shape[0] % N

            #openmp.omp_set_lock(&lock)
            #with gil: # Does not synchronize the prints ??
            #    print 'shape', shape_arg[0]
            #openmp.omp_unset_lock(&lock)

            try:
                function.invoke(shape_arg, data_pointers_copy, strides)
            finally:
                stdlib.free(data_pointers_copy)

        #openmp.omp_destroy_lock(&lock)
        return 0
