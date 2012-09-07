from numba import visitors as numba_visitors
from numba.minivect import (miniast,
                            minitypes,
                            specializers as minispecializers,
                            ctypes_conversion)
from numba import decorators, utils, functions

from numbapro import _internal
from numbapro.vectorize import _common, basic

import numpy as np

from libc cimport stdlib
cimport numpy as cnp

include "miniutils.pyx"

cdef extern from *:
    ctypedef int Py_intptr_t

ctypedef int minifunc(cnp.npy_intp *shape, void **data_pointers,
                      cnp.npy_intp **strides_pointers) except -1

ctypedef int unaryfunc(
        cnp.npy_intp *shape, void *, cnp.npy_intp *,
                             void *, cnp.npy_intp *) except -1
ctypedef int binaryfunc (
        cnp.npy_intp *shape, void *, cnp.npy_intp *,
                             void *, cnp.npy_intp *,
                             void *, cnp.npy_intp *) except -1
ctypedef int ternaryfunc(
        cnp.npy_intp *shape, void *, cnp.npy_intp *,
                             void *, cnp.npy_intp *,
                             void *, cnp.npy_intp *,
                             void *, cnp.npy_intp *) except -1


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

    def __init__(self, functions, nin):
        self.functions = functions
        self.nin = nin

    def __call__(self, *args, **kwds):
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
        key = arg_types + (broadcast.nd,)
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

        any_broadcasting = is_broadcasting(broadcast_args, broadcast)

        # Get the right specialization
        contig = (order & _internal.ARRAYS_ARE_CONTIG) and not any_broadcasting
        inner_contig = order & _internal.ARRAYS_ARE_INNER_CONTIG
        tiled = order & (_internal.ARRAYS_ARE_MIXED_CONTIG|
                         _internal.ARRAYS_ARE_MIXED_STRIDED)

        # contig_cfunc, inner_contig_cfunc, tiled_cfunc, strided_cfunc = ctypes_funcs
        (contig_func, inner_contig_func,
         tiled_func, strided_func) = function_pointers

        # print 'contig', contig, 'inner_contig', inner_contig, 'tiled', tiled
        if contig:
            # ctypes_func = contig_cfunc
            function_pointer = contig_func
        elif inner_contig:
            # ctypes_func = inner_contig_func
            function_pointer = inner_contig_func
        elif tiled:
            # ctypes_func = tiled_cfunc
            function_pointer = tiled_func
        else:
            # ctypes_func = strided_cfunc
            function_pointer = strided_func

        return self.run_ufunc(function_pointer, broadcast, ndim,
                              out, args, contig)

    cdef run_ufunc(self, Py_intptr_t function_pointer,
                         broadcast, int ndim, out, list arrays, bint contig):

        cdef cnp.npy_intp *shape_p, *strides_p
        cdef cnp.npy_intp **strides_args
        cdef void **data_pointers

        arrays.insert(0, out)
        broadcast_arrays(arrays, broadcast.shape, ndim, &shape_p, &strides_p)
        build_dynamic_args(arrays, strides_p, &data_pointers, &strides_args,
                           ndim)

        # For higher dimensional arrays we can still select a contiguous
        # specialization
        if contig and ndim > 3:
            for i in range(3, ndim):
                shape_p[0] *= shape_p[i]

        try:
            # self.dispatch(function_pointer, shape_p, strides_p, ndim, out, arrays)
            (<minifunc *> function_pointer)(shape_p, data_pointers, strides_args)
        finally:
            stdlib.free(shape_p)
            stdlib.free(strides_p)
            stdlib.free(data_pointers)
            stdlib.free(strides_args)

    cdef dispatch(self, Py_intptr_t function_pointer, cnp.npy_intp *shape_p,
                  cnp.npy_intp *strides_p, int ndim, out, list arrays):
        # TODO: modify minivect to read the data and stride pointers from a
        # TODO: C array
        cdef char *lhs_data = <char *> cnp.PyArray_DATA(out)
        cdef cnp.npy_intp *lhs_strides = &strides_p[0]
        cdef char *rhs_data = <char *> cnp.PyArray_DATA(arrays[1])
        cdef cnp.npy_intp *rhs_strides = &strides_p[ndim]

        if self.nin == 1:
            if ndim < 3:
                (<unaryfunc *> function_pointer)(shape_p, lhs_data, lhs_strides,
                                                          rhs_data, rhs_strides)
            else:
                self.unary_func_higher_dimensional(
                        <unaryfunc *> function_pointer, ndim, ndim, shape_p,
                        lhs_data, lhs_strides, rhs_data, rhs_strides)
        elif self.nin == 2:
            if ndim < 3:
                (<binaryfunc *> function_pointer)(
                        shape_p, lhs_data, lhs_strides, rhs_data, rhs_strides,
                        <char *> cnp.PyArray_DATA(arrays[2]),
                        &strides_p[ndim + ndim])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    cdef unary_func_higher_dimensional(
                self, unaryfunc *func, int ndim, int cur_ndim,
                cnp.npy_intp *shape,
                char *lhs_data, cnp.npy_intp *lhs_strides,
                char *rhs_data, cnp.npy_intp *rhs_strides):
        cdef int i

        if cur_ndim > 2:
            for i in range(shape[ndim - cur_ndim]):
                self.unary_func_higher_dimensional(
                    func, ndim, cur_ndim - 1, &shape[1],
                    lhs_data, &lhs_strides[1],
                    rhs_data, &rhs_strides[1])

                lhs_data += lhs_strides[0]
                rhs_data += rhs_strides[0]
        else:
            func(shape, <void *> lhs_data, lhs_strides,
                        <void *> rhs_data, rhs_strides)

    def run_ufunc_ctypes(self, ctypes_func, broadcast, ctypes_ret_type,
                         ctypes_arg_types, out, args):
        """
        Run the ufunc using ctypes
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
