from llvm.core import Type, inline_function
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm_cbuilder import builder

class SliceArray(CDefinition):

    _name_ = "slice"
    _retty_ = C.char_p
    _argtys_ = [
        ('data', C.char_p),
        ('shape', C.pointer(C.npy_intp)),
        ('strides', C.pointer(C.npy_intp)),
    ]

    def adjust_index(self, index, ):

    def body(self, data, shape, strides):
        start = self.start
        stop = self.stop
        step = self.step
        extent = shape[self.dimension]

        if start is not None:
            if start < 0:
                start += extent
                if start < 0:
                    start = 0
            elif start >= extent:
                if negative_step:
                    start = extent - 1
                else:
                    start = extent
        else:
            if negative_step:
                start = extent - 1
            else:
                start = 0

        if stop is not None:
            if stop < 0:
                stop += extent
                if stop < 0:
                    stop = 0
            elif stop > extent:
                stop = extent
        else:
            if negative_step:
                stop = -1
            else:
                stop = extent

        if step is None:
            step = 1

    @classmethod
    def specialize(cls, func_def, dimension, start, stop, step):
        cls.FuncDef = func_def
        cls.dimension = dimension
        cls.start = start
        cls.stop = stop
        cls.step = step