#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

import numba
from numba import compiler, typing
from numba.compiler import compile_isolated, Flags
from numba.targets import cpu
from numba import types
from numba.targets.registry import cpu_target
from numba import config
from numba.annotations import type_annotations
from numba.ir_utils import (copy_propagate, apply_copy_propagate,
                            get_name_var_table, remove_dels, remove_dead,
                            remove_call_handlers)
from numba import ir
from numba import unittest_support as unittest
import numpy as np
from .matmul_usecase import needs_blas


def test_will_propagate(b, z, w):
    x = 3
    if b > 0:
        y = z + w
    else:
        y = 0
    a = 2 * x
    return a < b

def null_func(a,b,c,d):
    False

def findLhsAssign(func_ir, var):
    for label, block in func_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, ir.Assign) and inst.target.name==var:
                return True

    return False

class TestRemoveDead(unittest.TestCase):
    def compile_parallel(self, func, arg_types):
        fast_pflags = Flags()
        fast_pflags.set('auto_parallel', cpu.ParallelOptions(True))
        fast_pflags.set('nrt')
        fast_pflags.set('fastmath')
        return compile_isolated(func, arg_types, flags=fast_pflags).entry_point

    def test1(self):
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx)
        test_ir = compiler.run_frontend(test_will_propagate)
        #print("Num blocks = ", len(test_ir.blocks))
        #print(test_ir.dump())
        with cpu_target.nested_context(typingctx, targetctx):
            typingctx.refresh()
            targetctx.refresh()
            args = (types.int64, types.int64, types.int64)
            typemap, return_type, calltypes = compiler.type_inference_stage(typingctx, test_ir, args, None)
            #print("typemap = ", typemap)
            #print("return_type = ", return_type)
            type_annotation = type_annotations.TypeAnnotation(
                func_ir=test_ir,
                typemap=typemap,
                calltypes=calltypes,
                lifted=(),
                lifted_from=None,
                args=args,
                return_type=return_type,
                html_output=config.HTML)
            remove_dels(test_ir.blocks)
            in_cps, out_cps = copy_propagate(test_ir.blocks, typemap)
            apply_copy_propagate(test_ir.blocks, in_cps, get_name_var_table(test_ir.blocks), typemap, calltypes)

            remove_dead(test_ir.blocks, test_ir.arg_names, test_ir)
            self.assertFalse(findLhsAssign(test_ir, "x"))

    def test2(self):
        def call_np_random_seed():
            np.random.seed(2)

        def seed_call_exists(func_ir):
            for inst in func_ir.blocks[0].body:
                if (isinstance(inst, ir.Assign) and
                    isinstance(inst.value, ir.Expr) and
                    inst.value.op == 'call' and
                    func_ir.get_definition(inst.value.func).attr == 'seed'):
                    return True
            return False

        test_ir = compiler.run_frontend(call_np_random_seed)
        remove_dead(test_ir.blocks, test_ir.arg_names, test_ir)
        self.assertTrue(seed_call_exists(test_ir))

    def run_array_index_test(self, func):
        A1 = np.arange(6).reshape(2,3)
        A2 = A1.copy()
        i = 0
        pfunc = self.compile_parallel(func, (numba.typeof(A1), numba.typeof(i)))

        func(A1, i)
        pfunc(A2, i)
        np.testing.assert_array_equal(A1, A2)

    def test_alias_ravel(self):
        def func(A, i):
            B = A.ravel()
            B[i] = 3

        self.run_array_index_test(func)

    def test_alias_flat(self):
        def func(A, i):
            B = A.flat
            B[i] = 3

        self.run_array_index_test(func)

    def test_alias_transpose1(self):
        def func(A, i):
            B = A.T
            B[i,0] = 3

        self.run_array_index_test(func)

    def test_alias_transpose2(self):
        def func(A, i):
            B = A.transpose()
            B[i,0] = 3

        self.run_array_index_test(func)

    def test_alias_transpose3(self):
        def func(A, i):
            B = np.transpose(A)
            B[i,0] = 3

        self.run_array_index_test(func)

    @needs_blas
    def test_alias_ctypes(self):
        # use xxnrm2 to test call a C function with ctypes
        from numba.targets.linalg import _BLAS
        xxnrm2 = _BLAS().numba_xxnrm2(types.float64)

        def remove_dead_xxnrm2(rhs, lives, call_list):
            if call_list == [xxnrm2]:
                return rhs.args[4].name not in lives
            return False

        # adding this handler has no-op effect since this function won't match
        # anything else but it's a bit cleaner to save the state and recover
        old_remove_handlers = remove_call_handlers[:]
        remove_call_handlers.append(remove_dead_xxnrm2)

        def func(ret):
            a = np.ones(4)
            xxnrm2(100, 4, a.ctypes, 1, ret.ctypes)

        A1 = np.zeros(1)
        A2 = A1.copy()

        try:
            pfunc = self.compile_parallel(func, (numba.typeof(A1),))
            numba.njit(func)(A1)
            pfunc(A2)
        finally:
            # recover global state
            remove_call_handlers[:] = old_remove_handlers

        self.assertEqual(A1[0], A2[0])

    def test_alias_reshape1(self):
        def func(A, i):
            B = np.reshape(A, (3,2))
            B[i,0] = 3

        self.run_array_index_test(func)

    def test_alias_reshape2(self):
        def func(A, i):
            B = A.reshape(3,2)
            B[i,0] = 3

        self.run_array_index_test(func)

if __name__ == "__main__":
    unittest.main()
