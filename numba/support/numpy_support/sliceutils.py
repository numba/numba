# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C

from numba.utility.cbuilder.library import register
from numba.utility.cbuilder.numbacdef import NumbaCDefinition, from_numba

def get_constants(cbuilder):
    zero = cbuilder.constant(C.npy_intp, 0)
    one = cbuilder.constant(C.npy_intp, 1)
    return one, zero

# @register
class SliceArray(CDefinition):

    _name_ = "slice"
    _retty_ = C.char_p
    _argtys_ = [
        ('data', C.char_p),

        ('in_shape', C.pointer(C.npy_intp)),
        ('in_strides', C.pointer(C.npy_intp)),

        ('out_shape', C.pointer(C.npy_intp)),
        ('out_strides', C.pointer(C.npy_intp)),

        ('start', C.npy_intp),
        ('stop', C.npy_intp),
        ('step', C.npy_intp),

        ('src_dim', C.int),
        ('dst_dim', C.int),
    ]

    def _adjust_given_index(self, extent, negative_step, index, is_start):
        # Tranliterate the below code to llvm cbuilder

        # TODO: write in numba

        # For the start index in start:stop:step, do:
        # if have_start:
        #     if start < 0:
        #         start += shape
        #         if start < 0:
        #             start = 0
        #     elif start >= shape:
        #         if negative_step:
        #             start = shape - 1
        #         else:
        #             start = shape
        # else:
        #     if negative_step:
        #         start = shape - 1
        #     else:
        #         start = 0

        # For the stop index, do:
        # if stop is not None:
        #     if stop < 0:
        #         stop += extent
        #         if stop < 0:
        #             stop = 0
        #     elif stop > extent:
        #         stop = extent
        # else:
        #     if negative_step:
        #         stop = -1
        #     else:
        #         stop = extent

        one, zero = get_constants(self)

        with self.ifelse(index < zero) as ifelse:
            with ifelse.then():
                index += extent
                with self.ifelse(index < zero) as ifelse_inner:
                    with ifelse_inner.then():
                        index.assign(zero)

            with ifelse.otherwise():
                with self.ifelse(index >= extent) as ifelse:
                    with ifelse.then():
                        if is_start:
                            # index is 'start' index
                            with self.ifelse(negative_step) as ifelse:
                                with ifelse.then():
                                    index.assign(extent - one)
                                with ifelse.otherwise():
                                    index.assign(extent)
                        else:
                            # index is 'stop' index. Stop is exclusive, to
                            # we don't care about the sign of the step
                            index.assign(extent)

    def _set_default_index(self, default1, default2, negative_step, index):
        with self.ifelse(negative_step) as ifelse:
            with ifelse.then():
                index.assign(default1)
            with ifelse.otherwise():
                index.assign(default2)

    def adjust_index(self, extent, negative_step, index, default1, default2,
                     is_start=False, have_index=True):
        if have_index:
            self._adjust_given_index(extent, negative_step, index, is_start)
        else:
            self._set_default_index(default1, default2, negative_step, index)

    def body(self, data, in_shape, in_strides, out_shape, out_strides,
             start, stop, step, src_dim, dst_dim):

        stride = in_strides[src_dim]
        extent = in_shape[src_dim]

        one, zero = get_constants(self)
        if not self.have_step:
            step = one

        negative_step = step < zero

        self.adjust_index(extent, negative_step, start,
                                  default1=extent - one, default2=zero,
                                  is_start=True, have_index=self.have_start)
        self.adjust_index(extent, negative_step, stop,
                                 default1=-one, default2=extent,
                                 have_index=self.have_stop)

        # self.debug("extent", extent)
        # self.debug("negative_step", negative_step.cast(C.npy_intp))
        # self.debug("start/stop/step", start, stop, step)
        new_extent = self.var(C.npy_intp)
        new_extent.assign((stop - start) / step)
        with self.ifelse((stop - start) % step != zero) as ifelse:
            with ifelse.then():
                new_extent += one

        with self.ifelse(new_extent < zero) as ifelse:
            with ifelse.then():
                new_extent.assign(zero)

        result = self.var(data.type, name='result')
        result.assign(data[start * stride:])
        out_shape[dst_dim] = new_extent
        # self.debug("new_extent", new_extent)
        # self.debug("out stride:", dst_dim, stride * step)
        out_strides[dst_dim] = stride * step

        self.ret(result)

    def specialize(self, context, have_start, have_stop, have_step):
        self.context = context

        self.have_start = have_start
        self.have_stop = have_stop
        self.have_step = have_step

        self._name_ = "slice_%s_%s_%s" % (have_start, have_stop, have_step)

@register
class IndexAxis(NumbaCDefinition):

    _name_ = "index"
    _retty_ = C.char_p
    _argtys_ = [
        ('data', C.char_p),
        ('in_shape', C.pointer(C.npy_intp)),
        ('in_strides', C.pointer(C.npy_intp)),
        ('src_dim', C.npy_intp),
        ('index', C.npy_intp),
    ]

    def body(self, data, in_shape, in_strides, src_dim, index):
        result = self.var(data.type, name='result')
        # self.debug("indexing...", src_dim, "stride", in_strides[src_dim])
        result.assign(data[in_strides[src_dim] * index:])
        self.ret(result)

@register
class NewAxis(NumbaCDefinition):

    _name_ = "newaxis"
    _argtys_ = [
        ('out_shape', C.pointer(C.npy_intp)),
        ('out_strides', C.pointer(C.npy_intp)),
        ('dst_dim', C.int),
    ]

    def body(self, out_shape, out_strides, dst_dim):
        one, zero = get_constants(self)
        out_shape[dst_dim] = one
        out_strides[dst_dim] = zero
        # self.debug("newaxis in dimension:", dst_dim)
        self.ret()

# TODO: Transliterate the below to a numba function

@register
class Broadcast(NumbaCDefinition):
    """
    Transliteration of

        @cname('__pyx_memoryview_broadcast')
        cdef bint __pyx_broadcast(Py_ssize_t *dst_shape,
                                  Py_ssize_t *input_shape,
                                  Py_ssize_t *strides,
                                  int max_ndim, int ndim,
                                  bint *p_broadcast) nogil except -1:
            cdef Py_ssize_t i
            cdef int dim_offset = max_ndim - ndim

            for i in range(ndim):
                src_extent = input_shape[i]
                dst_extent = dst_shape[i + dim_offset]

                if src_extent == 1:
                    p_broadcast[0] = True
                    strides[i] = 0
                elif dst_extent == 1:
                    dst_shape[i + dim_offset] = src_extent
                elif src_extent != dst_extent:
                    __pyx_err_extents(i, dst_shape[i], input_shape[i])
    """

    _name_ = "__numba_util_broadcast"
    _argtys_ = [
        ('dst_shape', C.pointer(C.npy_intp)),
        ('src_shape', C.pointer(C.npy_intp)),
        ('src_strides', C.pointer(C.npy_intp)),
        ('max_ndim', C.int),
        ('ndim', C.int),
    ]
    _retty_ = C.int

    def body(self, dst_shape, src_shape, src_strides, max_ndim, ndim):
        dim_offset = max_ndim - ndim

        def constants(type):
            return self.constant(type, 0), self.constant(type, 1)

        zero, one = constants(C.npy_intp)
        zero_int, one_int = constants(C.int)

        with self.for_range(ndim) as (loop, i):
            src_extent = src_shape[i]
            dst_extent = dst_shape[i + dim_offset]

            with self.ifelse(src_extent == one) as ifelse:
                with ifelse.then():
                    src_strides[i] = zero
                with ifelse.otherwise():
                    with self.ifelse(dst_extent == one) as ifelse:
                        with ifelse.then():
                            dst_shape[i + dim_offset] = src_extent

                        with ifelse.otherwise():
                            with self.ifelse(src_extent != dst_extent) as ifelse:
                                with ifelse.then():
                                    # Shape mismatch
                                    self.ret(zero_int)

        self.ret(one_int)
