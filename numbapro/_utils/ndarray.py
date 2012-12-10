import numpy as np

def ndarray_datasize(ary):
    if not ary.strides[0]: # zero strides
        datasize = ary.dtype.itemsize
    elif ary.flags['C_CONTIGUOUS']:
        assert ary.strides[0] != 0
        datasize = ary.shape[0] * ary.strides[0]
    elif ary.flags['F_CONTIGUOUS']:
        assert ary.strides[-1] != 0
        datasize = ary.shape[-1] * ary.strides[-1]
    else:
        raise Exception("Array is neither C/F contiguous")
    return datasize