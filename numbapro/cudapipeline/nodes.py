import numpy

from numba import nodes
from numba.symtab import Variable
from numba.minivect import minitypes

class CudaAttributeNode(nodes.Node):
    _attributes = ['value']

    def __init__(self, value):
        self.value = value

    def resolve(self, name):
        return type(self)(getattr(self.value, name))

    def __repr__(self):
        return '<%s value=%s>' % (type(self).__name__, self.value)

class CudaSMemArrayNode(nodes.Node):
    pass

class CudaSMemArrayCallNode(nodes.Node):
    _attributes = ('shape', 'variable')
    def __init__(self, context, shape, dtype):

        self.shape = shape
        tmp_strides = [dtype.itemsize]
        for s in reversed(self.shape[1:]):
            tmp_strides.append(tmp_strides[-1] * s)
        self.strides = tuple(reversed(tmp_strides))

        self.elemcount = numpy.prod(self.shape)
        self.dtype = dtype
        type = minitypes.ArrayType(dtype=dtype,
                                   ndim=len(self.shape),
                                   is_c_contig=True)

        self.variable = Variable(type, promotable_type=False)


class CudaSMemAssignNode(nodes.Node):

    _fields = ['target', 'value']

    def __init__(self, target, value):
        self.target = target
        self.value = value

class CudaMacroGridNode(nodes.Node):
    pass

class CudaMacroGridExpandValuesNode(nodes.Node):
    pass

