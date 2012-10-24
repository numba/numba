#! /usr/bin/env python
# ______________________________________________________________________
'''test_multiarray_api

Test the code generation utility class numba.multiarray_api.MultiarrayAPI.
'''
# ______________________________________________________________________

import ctypes

import llvm.core as lc
import llvm.ee as le

from numba.llvm_types import _int32, _intp, _intp_star, _void_star, _numpy_array, _head_len
import numba.multiarray_api as ma

import numpy as np

import unittest

import logging
logging.basicConfig(level=logging.DEBUG)

# ______________________________________________________________________

_pyobj_to_pyobj = ctypes.CFUNCTYPE(ctypes.py_object, ctypes.py_object)


_void_star = lc.Type.pointer(lc.Type.int(8))

_make_const_int = lambda X: lc.Constant.int(_int32, X)

def _numpy_array_element(ndarray_ptr, idx, builder):
    ptr_to_element = builder.gep(ndarray_ptr,
                                 map(_make_const_int, [0, _head_len + idx]))
    return builder.load(ptr_to_element)

def _get_pyarray_getptr(module):
    '''
    For reference:

        void *
        PyArray_GetPtr(PyArrayObject *obj, npy_intp* ind)
        {
            int n = PyArray_NDIM(obj);
            npy_intp *strides = PyArray_STRIDES(obj);
            char *dptr = PyArray_DATA(obj);

            while (n--) {
                dptr += (*strides++) * (*ind++);
            }
            return (void *)dptr;
        }
    '''
    function_type = lc.Type.function(_void_star, [_numpy_array, _intp_star])
    function = module.get_or_insert_function(function_type, 'PyArray_GetPtr_inline')
    if function.basic_block_count!=0:
        # Already implemented in the module
        return function

    # set linkage and attributes
    function.add_attribute(lc.ATTR_ALWAYS_INLINE) # force inline
    function.linkage = lc.LINKAGE_INTERNAL

    # implement the function
    bb_entry = function.append_basic_block('entry')
    bb_while_cond = function.append_basic_block('while.cond')
    bb_while_body = function.append_basic_block('while.body')
    bb_ret = function.append_basic_block('return')

    # initialize
    builder = lc.Builder.new(bb_entry)

    ndarray_element = lambda X: _numpy_array_element(ndarray_ptr, X, builder)

    ndarray_ptr = function.args[0]
    dptr, nd, strides = map(ndarray_element, [0, 1, 3])
    dptr.name = 'dptr'
    nd.name = 'nd'
    strides.name = 'strides'

    builder.branch(bb_while_cond)               # branch to while cond

    # while (n--)
    builder.position_at_end(bb_while_cond)

    nd_phi = builder.phi(nd.type, name='nd_phi')
    nd_phi.add_incoming(nd, bb_entry)

    nd_minus_one = builder.sub(nd_phi, _make_const_int(1), name='nd_minus_one')
    nd_phi.add_incoming(nd_minus_one, bb_while_body)

    pred = builder.icmp(lc.ICMP_NE, nd_phi, _make_const_int(0))

    strides_phi = builder.phi(strides.type, name='strides_phi')
    strides_phi.add_incoming(strides, bb_entry)

    ind_phi = builder.phi(function.args[1].type, name='ind_phi')
    ind_phi.add_incoming(function.args[1], bb_entry)

    dptr_phi = builder.phi(dptr.type, name='dptr_phi')
    dptr_phi.add_incoming(dptr, bb_entry)

    builder.cbranch(pred, bb_while_body, bb_ret)

    # dptr += (*strides++) * (*ind++);
    builder.position_at_end(bb_while_body)

    strides_next = builder.gep(strides_phi, [_make_const_int(1)], name='strides_next')
    strides_phi.add_incoming(strides_next, bb_while_body)

    ind_next = builder.gep(ind_phi, [_make_const_int(1)], name='ind_next')
    ind_phi.add_incoming(ind_next, bb_while_body)

    stride_value = builder.load(strides_phi)
    ind_value = builder.load(builder.bitcast(ind_phi, strides_phi.type))
    dptr_next = builder.gep(dptr_phi, [builder.mul(stride_value, ind_value)], name='dptr_next')
    dptr_phi.add_incoming(dptr_next, bb_while_body)

    builder.branch(bb_while_cond)

    # return (void *) dptr;
    builder.position_at_end(bb_ret)

    builder.ret(dptr_phi)

    # check generated code
    function.verify()

    return function

# ______________________________________________________________________

# For reference:
#    typedef struct {
#    PyObject_HEAD                   // indices (skipping the head)
#    char *data;                     // 0
#    int nd;                         // 1
#    int *dimensions, *strides;      // 2, 3
#    PyObject *base;                 // 4
#    PyArray_Descr *descr;           // 5
#    int flags;                      // 6
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

        logging.debug(module)

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
        This is not the recommended way to access elements in ndarray.
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

        # prepare arg 1 PyObject** op
        pyobj_ptr = builder.alloca(arg_pyobj.type)
        builder.store(arg_pyobj, pyobj_ptr)
        arg_pyobj_ptr = builder.bitcast(pyobj_ptr, lc.Type.pointer(_void_star))

        # prepare arg 2 void* ptr

        data_ptr = builder.alloca(lc.Type.pointer(lc.Type.double()))
        arg_data_ptr = builder.bitcast(data_ptr, _void_star)

        # prepare arg 3, 4, 5
        ndarray_element = lambda X: _numpy_array_element(arg_pyobj, X, builder)
        nd, dimensions, descr = map(ndarray_element, [1, 2, 5])

        descr_as_void_ptr = builder.bitcast(descr, _void_star)

        # call
        largs = [arg_pyobj_ptr, arg_data_ptr, dimensions, nd, descr_as_void_ptr]


        status = builder.call(pyarray_ascarray, largs)
        # check errors?
        #    builder.ret(status)


        data_array = builder.load(data_ptr)

        data = []
        for i in xrange(3): # The count is fixed for this simple test.
            elem_ptr = builder.gep(data_array, map(_make_const_int, [i]))
            data.append(builder.load(elem_ptr))

        sum_data = builder.fadd(builder.fadd(data[0], data[1]), data[2])
        builder.ret(sum_data)

        # NOTE: The arg_data_ptr is never freed. This is okay only for test here.

        logging.debug(module)

        test_fn.verify()

        ee = le.ExecutionEngine.new(module)
        test_fn_addr = ee.get_pointer_to_function(test_fn)

        c_func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.py_object)
        py_test_fn = c_func_type(test_fn_addr)

        test_arr = np.array([1.234, 2.345, 3.567])
        result = py_test_fn(test_arr)

        self.assertEqual(sum(test_arr), result)

    def test_call_PyArray_GetPtr(self):
        '''
        Using PyArray_GetPtr should be the preferred method to access the
        element. The only thing I am concerning is we will miss optimization
        opportunity since LLVM has no information of PyArray_GetPtr. Perhaps,
        It is better to put the definition inside the LLVM module.
        '''

        ma_obj = ma.MultiarrayAPI()
        module = lc.Module.new('test_module_PyArray_GetPtr')
        ma_obj.set_PyArray_API(module)
        test_fn = module.add_function(lc.Type.function(lc.Type.double(),
                                                       [_numpy_array, _int32]),
                                      'test_fn')
        bb = test_fn.append_basic_block('entry')
        builder = lc.Builder.new(bb)
        pyarray_getptr = ma_obj.load_PyArray_GetPtr(module, builder)
        pyarray_getptr_fnty = pyarray_getptr.type.pointee

        # prepare arg 1 PyObject *
        arg_pyobj = test_fn.args[0]

        npy_intp_ty = pyarray_getptr_fnty.args[1].pointee

        # prepare arg 2 npy_intp *
        arg_index = builder.alloca(npy_intp_ty)
        index_as_npy_intp = builder.sext(test_fn.args[1], npy_intp_ty)
        builder.store(index_as_npy_intp, arg_index)

        # call
        largs = [arg_pyobj, arg_index]
        elemptr = builder.call(pyarray_getptr, largs)

        # return the loaded element at the index specified
        elemptr_as_double_ptr = builder.bitcast(elemptr, lc.Type.pointer(lc.Type.double()))
        builder.ret(builder.load(elemptr_as_double_ptr))

        logging.debug(module)

        test_fn.verify()

        ee = le.ExecutionEngine.new(module)
        test_fn_addr = ee.get_pointer_to_function(test_fn)

        c_func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.py_object, ctypes.c_int)
        py_test_fn = c_func_type(test_fn_addr)

        test_arr = np.array([1.234, 2.345, 3.567])

        for idx, val in enumerate(test_arr):
            result = py_test_fn(test_arr, idx)
            self.assertEqual(val, result)

    def test_call_PyArray_GetPtr_inline(self):
        '''
        Let's try implementing PyArray_GetPtr inside LLVM and allow inlining.
        '''
        module = lc.Module.new('test_module_PyArray_GetPtr_inline')


        test_fn = module.add_function(lc.Type.function(lc.Type.double(),
                                                       [_numpy_array, _int32]),
                                      'test_fn')
        bb = test_fn.append_basic_block('entry')
        builder = lc.Builder.new(bb)
        pyarray_getptr = _get_pyarray_getptr(module)
        pyarray_getptr_fnty = pyarray_getptr.type.pointee

        # prepare arg 1 PyObject *
        arg_pyobj = test_fn.args[0]

        npy_intp_ty = pyarray_getptr_fnty.args[1].pointee

        # prepare arg 2 npy_intp *
        arg_index = builder.alloca(npy_intp_ty)
        index_as_npy_intp = builder.sext(test_fn.args[1], npy_intp_ty)
        builder.store(index_as_npy_intp, arg_index)

        # call
        largs = [arg_pyobj, arg_index]
        elemptr = builder.call(pyarray_getptr, largs)

        # return the loaded element at the index specified
        elemptr_as_double_ptr = builder.bitcast(elemptr, lc.Type.pointer(lc.Type.double()))
        builder.ret(builder.load(elemptr_as_double_ptr))

        logging.debug(module)

        test_fn.verify()

        ee = le.ExecutionEngine.new(module)
        test_fn_addr = ee.get_pointer_to_function(test_fn)

        c_func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.py_object, ctypes.c_int)
        py_test_fn = c_func_type(test_fn_addr)

        test_arr = np.array([1.234, 2.345, 3.567])

        for idx, val in enumerate(test_arr):
            result = py_test_fn(test_arr, idx)
            self.assertEqual(val, result)


# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_multiarray_api.py
