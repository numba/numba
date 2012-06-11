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
            for ofs in (1, 2, 5)]
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

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_multiarray_api.py
