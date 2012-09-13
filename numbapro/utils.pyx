cimport cython.parallel
from libc cimport stdlib, string as libc_string
cimport numpy as cnp
cimport utils as cutils

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

    def key(self, arg_dtypes, broadcast):
        return arg_dtypes, broadcast.nd

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
        key = self.key(arg_types, broadcast)
        if key not in self.functions:
            raise ValueError(
                "No signature found for input types (%s, ndim=%d)" % (
                    ", ".join(str(dtype) for dtype in arg_types), broadcast.nd))

        #(functions, ctypes_funcs, ctypes_ret_type,
        # ctypes_arg_types, result_dtype) = self.functions[key]
        functions, result_dtype = self.functions[key]

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

        contig_func, inner_contig_func, tiled_func, strided_func = functions

        # print 'contig', contig, 'inner_contig', inner_contig, 'tiled', tiled
        if contig:
            function = contig_func
        elif inner_contig:
            function = inner_contig_func
        # elif tiled:
            # function = tiled_func
        else:
            function = strided_func

        return self.run_ufunc(function, broadcast, ndim, out, args, contig)

    cdef run_ufunc(self, Function function, broadcast, int ndim, out,
                   list arrays, bint contig):
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
        if contig and ndim > self.max_specialization_ndim and not self.parallel:
            for i in range(self.max_specialization_ndim, ndim):
                shape_p[0] *= shape_p[i]

        try:
            if (self.parallel and ndim > self.max_specialization_ndim and not
                    contig):
                # TODO: map 2D arrays to parallel 1D arrays? Might be bad, e.g.
                # TODO: can't tile... Better implement thread-pool in minivect
                self.run_higher_dimensional_parallel(
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
                            function, shape, data_pointers_copy,
                            strides_pointers_copy, ndim - 1, n_ops,
                            dim_level + 1, contig)
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
