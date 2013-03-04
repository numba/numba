import numpy as np

def ndarray_datasize(ary):
    order = ''
    if ary.flags['C_CONTIGUOUS']:
        order = 'C'
    elif ary.flags['F_CONTIGUOUS']:
        order = 'F'
    return ndarray_datasize_raw(shape = ary.shape,
                                strides = ary.strides,
                                dtype = ary.dtype,
                                order = order)


def ndarray_datasize_raw(shape, strides, dtype, order):
    if not strides[0]: # zero strides
        datasize = dtype.itemsize
    elif order == 'C':
        assert strides[0] != 0
        datasize = shape[0] * strides[0]
    elif order == 'F':
        assert strides[-1] != 0
        datasize = shape[-1] * strides[-1]
    else:
        raise Exception("Array is neither C/F contiguous")
    return datasize

