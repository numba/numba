cimport cython.parallel

from libc cimport stdlib, string as libc_string
cimport numpy as cnp


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

cdef extern from *:
    ctypedef int Py_intptr_t

ctypedef int minifunc(cnp.npy_intp *shape, char **data_pointers,
                      cnp.npy_intp **strides_pointers) nogil except -1

DEF MAX_SPECIALIZATION_NDIM = 2

cdef class UFuncDispatcher(object):
    """
    Given a dict of signatures mapping to functions, dispatch to the right
    ufunc for element-wise operations.

    functions: a dict mapping (dtypeA, dtypeB, dimensionality) ->
                                    (func_pointer, ctypes_func, ctypes_ret_type,
                                     ctypes_arg_type, result_dtype)
    """

    cdef object functions
    cdef int nin
    cdef bint parallel

    def __init__(self, functions, nin, parallel):
        self.functions = functions
        self.nin = nin
        self.parallel = parallel

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

        # Find ufunc from input arrays
        key = arg_types + (min(broadcast.nd, 2),)
        if key not in self.functions:
            raise ValueError("No signature found for %s" % (arg_types,))

        (function_pointers, ctypes_funcs, ctypes_ret_type,
                            ctypes_arg_types, result_dtype) = self.functions[key]

        # Instantiate the LHS
        if out is None:
            if order & _internal.ARRAY_C_ORDER:
                order_arg = 'C'
            else:
                order_arg = 'F'

            out = np.empty(broadcast.shape, dtype=result_dtype, order=order_arg)
        else:
            # TODO: check for read-after-write
            pass

        any_broadcasting = is_broadcasting(broadcast_args, broadcast)

        # Get the right specialization
        contig = (order & _internal.ARRAYS_ARE_CONTIG) and not any_broadcasting
        inner_contig = order & _internal.ARRAYS_ARE_INNER_CONTIG
        tiled = order & (_internal.ARRAYS_ARE_MIXED_CONTIG|
                         _internal.ARRAYS_ARE_MIXED_STRIDED)

        # contig_cfunc, inner_contig_cfunc, tiled_cfunc, strided_cfunc = ctypes_funcs
        (contig_func, inner_contig_func, tiled_func,
         strided_func) = function_pointers

        # print 'contig', contig, 'inner_contig', inner_contig, 'tiled', tiled
        if contig:
            # ctypes_func = contig_cfunc
            function_pointer = contig_func
        elif inner_contig:
            # ctypes_func = inner_contig_func
            function_pointer = inner_contig_func
        # elif tiled:
            # ctypes_func = tiled_cfunc
            # function_pointer = tiled_func
        else:
            # ctypes_func = strided_cfunc
            function_pointer = strided_func

        return self.run_ufunc(function_pointer, broadcast, ndim,
                              out, args, contig)

    cdef run_ufunc(self, Py_intptr_t function_pointer,
                         broadcast, int ndim, out, list arrays, bint contig):
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

        # For higher dimensional arrays we can still select a contiguous
        # specialization in a single go
        if contig and ndim > MAX_SPECIALIZATION_NDIM and not self.parallel:
            for i in range(MAX_SPECIALIZATION_NDIM, ndim):
                shape_p[0] *= shape_p[i]

        try:
            if self.parallel and ndim > MAX_SPECIALIZATION_NDIM and not contig:
                # TODO: map 2D arrays to parallel 1D arrays? Might be bad, e.g.
                # TODO: can't tile... Better implement thread-pool in minivect
                self.run_higher_dimensional_parallel(
                        <minifunc *> function_pointer, shape_p,
                        data_pointers, strides_args, ndim,
                        len(arrays), 0)
            else:
                self.run_higher_dimensional(
                        <minifunc *> function_pointer, shape_p,
                        data_pointers, strides_args, ndim,
                        len(arrays), 0)
        finally:
            stdlib.free(shape_p)
            stdlib.free(strides_p)
            stdlib.free(data_pointers)
            stdlib.free(strides_args)

        return arrays[0]

    cdef int run_higher_dimensional(
            self, minifunc *function_pointer, cnp.npy_intp *shape,
            char **data_pointers, cnp.npy_intp **strides,
            int ndim, int n_ops, int dim_level) nogil except -1:
        """
        Run the 1 or 2 dimensional ufunc. If ndim > 2, we need a to simulate
        an outer loop nest of depth ndim - 2.
        """
        cdef int i, j

        if ndim <= MAX_SPECIALIZATION_NDIM:
            # Offset the stride pointer to remove the strides for
            # preceding dimensions
            # TODO: have minivect accept a dim_level argument
            if dim_level:
                for j in range(n_ops):
                    strides[j] += dim_level

            function_pointer(&shape[dim_level], data_pointers, strides)

            # Reset the stride pointers to the first stride for each operand
            if dim_level:
                for j in range(n_ops):
                    strides[j] -= dim_level
        else:
            # ndim > 2
            for i in range(shape[dim_level]):
                self.run_higher_dimensional(function_pointer, shape,
                                            data_pointers, strides, ndim - 1,
                                            n_ops, dim_level + 1)

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
            self, minifunc *function_pointer, cnp.npy_intp *shape,
            char **data_pointers, cnp.npy_intp **strides,
            int ndim, int n_ops, int dim_level) nogil except -1:
        """
        Run the ufunc using OpenMP on the outermost loop level.
        """
        cdef int i, j

        cdef char **data_pointers_copy
        cdef cnp.npy_intp **strides_pointers_copy

        with nogil, cython.parallel.parallel():
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
                            function_pointer, shape, data_pointers_copy,
                            strides_pointers_copy, ndim - 1, n_ops,
                            dim_level + 1)
            finally:
                stdlib.free(data_pointers_copy)
                stdlib.free(strides_pointers_copy)

        return 0

    def run_ufunc_ctypes(self, ctypes_func, broadcast, ctypes_ret_type,
                         ctypes_arg_types, out, args):
        """
        Run the ufunc using ctypes. Unused.
        """
        # Create broadcasted shape argument
        ctypes_shape_array = args[0].ctypes.shape._type_ * broadcast.nd
        ctypes_shape_array = ctypes_shape_array(*broadcast.shape)
        call_args = [ctypes_shape_array]

        def add_arg(array, ctypes_type):
            call_args.append(array.ctypes.data_as(ctypes_type))
            #if not contig:
            if array.ndim == broadcast.nd:
                strides = array.ctypes.strides
            else:
                # handle lesser dimensional operands
                # (prepend broadcasting axes)
                strides = ctypes_type * broadcast.nd
                leading_dims = broadcast.nd - array.ndim
                strides = strides(leading_dims * (0,) + array.strides)

            call_args.append(strides)

        add_arg(out, ctypes_ret_type)

        # Get stride arguments
        # TODO: handle lesser dimensional broadcasting operands
        for numpy_array, ctypes_arg_type in zip(args, ctypes_arg_types):
            add_arg(numpy_array, ctypes_arg_type)

        # Run the ufunc!
        ctypes_func(*call_args)
        return out
