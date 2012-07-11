#! /usr/bin/env python
# ______________________________________________________________________
'''test_multiarray_api

Test the code generation utility class numba.multiarray_api.MultiarrayAPI.
'''
# ______________________________________________________________________

import ctypes

import llvm.core as lc
import llvm.ee as le

from numba.llvm_types import _int32, _numpy_array, _head_len
import numba.multiarray_api as ma

import numpy as np

import unittest

# ______________________________________________________________________

_pyobj_to_pyobj = ctypes.CFUNCTYPE(ctypes.py_object, ctypes.py_object)

# ______________________________________________________________________

# For reference:
#    typedef struct {
#    PyObject_HEAD
#    char *data;
#    int nd;
#    int *dimensions, *strides;
#    PyObject *base;
#    PyArray_Descr *descr;
#    int flags;
#    } PyArrayObject;

class TestMultiarrayAPI(unittest.TestCase):
    def test_call_PyArray_Zeros(self):
        ma_obj = ma.MultiarrayAPI()
        module = lc.Module.new('test_module')
        ma_obj.set_PyArray_API(module)
        test_fn = module.add_function(lc.Type.function(_numpy_array,
                                                       [_numpy_array]),
                                      'test_fn')
        bb = test_fn.append_basic_block('entry')
        builder = lc.Builder.new(bb)
        pyarray_zeros = ma_obj.load_PyArray_Zeros(module, builder)
        arg = test_fn.args[0]
        largs = [
            builder.load(
                builder.gep(arg,
                            [lc.Constant.int(_int32, 0),
                             lc.Constant.int(_int32, _head_len + ofs)]))
            for ofs in (1, 2, 5)] # nd, dimensions, descr
        largs.append(lc.Constant.int(_int32, 0))
        builder.ret(builder.call(pyarray_zeros, largs))
        if __debug__:
            print module
        ee = le.ExecutionEngine.new(module)
        test_fn_addr = ee.get_pointer_to_function(test_fn)
        py_test_fn = _pyobj_to_pyobj(test_fn_addr)
        test_arr = np.array([1.,2.,3.])
        result = py_test_fn(test_arr)
        self.assertEqual(result.shape, test_arr.shape)
        self.assertTrue((result == 0.).all())

    def test_call_PyArray_AsCArray(self):
        '''
        A test to check PyArray_AsCArray for accessing the C-array in ndarray.
        This will also serve as a guide to implement numpy array access in
        the codegen.
        '''
        ma_obj = ma.MultiarrayAPI()
        module = lc.Module.new('test_module_PyArray_AsCArray')
        ma_obj.set_PyArray_API(module)
        test_fn = module.add_function(lc.Type.function(lc.Type.double(), #_numpy_array,
                                                       [_numpy_array]),
                                      'test_fn')
        bb = test_fn.append_basic_block('entry')
        builder = lc.Builder.new(bb)
        pyarray_ascarray = ma_obj.load_PyArray_AsCArray(module, builder)
        pyarray_ascarray_fnty = pyarray_ascarray.type.pointee

        arg_pyobj = test_fn.args[0]

        void_ptr_ty = lc.Type.pointer(lc.Type.int(8))

        make_const_int = lambda X: lc.Constant.int(_int32, X)

        # prepare arg 1 PyObject** op
        pyobj_ptr = builder.alloca(arg_pyobj.type)
        builder.store(arg_pyobj, pyobj_ptr)
        arg_pyobj_ptr = builder.bitcast(pyobj_ptr, lc.Type.pointer(void_ptr_ty))

        # prepare arg 2 void* ptr

        data_ptr = builder.alloca(lc.Type.pointer(lc.Type.double()))
        arg_data_ptr = builder.bitcast(data_ptr, void_ptr_ty)

        # prepare arg 3, 4, 5
        nd, dimensions, descr = [
            builder.load(
                builder.gep(arg_pyobj,
                            map(make_const_int, [0, _head_len + ofs])))
            for ofs in (1, 2, 5)] # nd, dimensions, descr

        descr_as_void_ptr = builder.bitcast(descr, void_ptr_ty)

        # call
        largs = [arg_pyobj_ptr, arg_data_ptr, dimensions, nd, descr_as_void_ptr]

        status = builder.call(pyarray_ascarray, largs)
        # check errors?
        #    builder.ret(status)

        data_array = builder.load(data_ptr)

        data = []
        for i in xrange(3): # The count is fixed for this simple test.
            elem_ptr = builder.gep(data_array, map(make_const_int, [i]))
            data.append(builder.load(elem_ptr))

        sum_data = builder.fadd(builder.fadd(data[0], data[1]), data[2])
        builder.ret(sum_data)

        if __debug__:
            print module

        test_fn.verify()
        module.verify()

        ee = le.ExecutionEngine.new(module)
        test_fn_addr = ee.get_pointer_to_function(test_fn)

        c_func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.py_object)
        py_test_fn = c_func_type(test_fn_addr)

        test_arr = np.array([1.234, 2.345, 3.567])
        result = py_test_fn(test_arr)

        self.assertEqual(sum(test_arr), result)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_multiarray_api.py
