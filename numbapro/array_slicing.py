from llvm.core import Type, inline_function
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm_cbuilder import builder

class SliceArray(CDefinition):

    _name_ = "slice"
    _retty_ = C.char_p
    _argtys_ = [
        ('data', C.char_p),
        ('in_shape', C.pointer(C.npy_intp)),
        ('in_strides', C.pointer(C.npy_intp)),
        ('out_shape', C.pointer(C.npy_intp)),
        ('out_strides', C.pointer(C.npy_intp))
    ]


    def _adjust_given_index(self, extent, negative_step, index, is_start):
        # Tranliterate the below code to llvm cbuilder

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

        with self.parent.ifelse(index < 0) as ifelse:
            with ifelse.then():
                index += extent
                if index < 0:
                    index.assign(0)

            with ifelse.otherwise():
                with self.parent.ifelse(index >= extent) as ifelse:
                    if is_start:
                        # index is 'start' index
                        with self.parent.ifelse(negative_step) as ifelse:
                            with ifelse.then():
                                index.assign(extent - 1)
                            with ifelse.otherwise():
                                index.assign(extent)
                    else:
                        # index is 'stop' index. Stop is exclusive, to
                        # we don't care about the sign of the step
                        index.assign(extent)

    def _set_default_index(self, default1, default2, negative_step, start):
        with self.parent.ifelse(negative_step) as ifelse:
            with ifelse.then():
                start.assign(default1)
            with ifelse.otherwise():
                start.assign(default2)

    def adjust_index(self, extent, negative_step, index, default1, default2,
                     is_start=False):
        if index is not None:
            self._adjust_given_index(extent, negative_step, index, is_start)
        else:
            index = self.parent.var(C.npy_intp)
            self._set_default_index(default1, default2, negative_step, index)

        return index

    def body(self, data, in_shape, in_strides, out_shape, out_strides):
        start = self.start
        stop = self.stop
        step = self.step

        dim = self.src_dimension
        stride = strides[dim]
        extent = shape[dim]

        negative_step = step < 0

        start = self.adjust_index(extent, negative_step, start, extent - 1, 0,
                                  is_start=True)
        stop = self.adjust_index(extent, negative_step, start, -1, extent)
        if step is None:
            step = 1

        new_extent = (stop - start) / step
        with self.parent.ifelse((stop - start) % step != 0) as ifelse:
            with ifelse.then():
                new_extent += 1
        with self.parent.ifelse(new_shape < 0) as ifelse:
            with ifelse.then():
                new_shape.assign(0)

        dst_dim = self.dst_dimension

        result = self.parent.var(data.type, name='result')
        result.assign(data + start * stride)
        shape[dim] = new_extent
        strides[dim] = stride * step

        self.ret(result)

    @classmethod # specializing classes is bad, ameliorate
    def specialize(cls, src_dimension, dst_dimension, start, stop, step):
        cls.FuncDef = func_def
        cls.src_dimension = src_dimension
        cls.dst_dimension = dst_dimension
        cls.start = start
        cls.stop = stop
        cls.step = step

class IndexAxis(CDefinition):

    _name_ = "index"
    _retty_ = C.char_p
    _argtys_ = [
        ('data', C.char_p),
        ('in_shape', C.pointer(C.npy_intp)),
        ('in_strides', C.pointer(C.npy_intp)),
        ('index', C.npy_intp),
        ('src_dim', C.npy_intp),
    ]

    def body(self, data, in_shape, in_strides, index, src_dim):
        result = self.parent.var(data.type, name='result')
        result.assign(data + in_strides[src_dim] * index)
        self.ret(result)

    @classmethod
    def specialize(cls):
        pass

class NewAxis(CDefinition):

    _name_ = "newaxis"
    _argtys_ = [
        ('out_shape', C.pointer(C.npy_intp)),
        ('out_strides', C.pointer(C.npy_intp)),
        ('dst_dim', C.npy_intp),
    ]

    def body(self, out_shape, out_strides, dst_dim):
        out_shape[dst_dim] = 1
        out_strides[dst_dim] = 0

    @classmethod
    def specialize(cls):
        pass
