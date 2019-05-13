"""
Unspecified error handling tests
"""
from __future__ import division

from numba import jit, njit
from numba import unittest_support as unittest
from numba import errors, utils
import numpy as np


class TestErrorHandlingBeforeLowering(unittest.TestCase):

    expected_msg = ("Numba encountered the use of a language feature it does "
                    "not support in this context: %s")

    def test_unsupported_make_function_lambda(self):
        def func(x):
            f = lambda x: x  # requires `make_function`

        for pipeline in jit, njit:
            with self.assertRaises(errors.UnsupportedError) as raises:
                pipeline(func)(1)

            expected = self.expected_msg % "<lambda>"
            self.assertIn(expected, str(raises.exception))

    def test_unsupported_make_function_return_inner_func(self):
        def func(x):
            """ return the closure """
            z = x + 1

            def inner(x):
                return x + z
            return inner

        for pipeline in jit, njit:
            with self.assertRaises(errors.UnsupportedError) as raises:
                pipeline(func)(1)

            expected = self.expected_msg % \
                "<creating a function from a closure>"
            self.assertIn(expected, str(raises.exception))


class TestUnsupportedReporting(unittest.TestCase):

    def test_unsupported_numpy_function(self):
        # np.asanyarray(list) currently unsupported
        @njit
        def func():
            np.asanyarray([1,2,3])

        with self.assertRaises(errors.TypingError) as raises:
            func()

        expected = "Use of unsupported NumPy function 'numpy.asanyarray'"
        self.assertIn(expected, str(raises.exception))


class TestMiscErrorHandling(unittest.TestCase):

    def test_use_of_exception_for_flow_control(self):
        # constant inference uses exceptions with no Loc specified to determine
        # flow control, this asserts that the construction of the lowering
        # error context handler works in the case of an exception with no Loc
        # specified. See issue #3135.
        @njit
        def fn(x):
            return 10**x

        a = np.array([1.0],dtype=np.float64)
        fn(a) # should not raise

    def test_use_of_ir_unknown_loc(self):
        # for context see # 3390
        import numba
        class TestPipeline(numba.compiler.BasePipeline):
            def define_pipelines(self, pm):
                pm.create_pipeline('test_loc')
                self.add_preprocessing_stage(pm)
                self.add_with_handling_stage(pm)
                self.add_pre_typing_stage(pm)
                # remove dead before type inference so that the Arg node is removed
                # and the location of the arg cannot be found
                pm.add_stage(self.rm_dead_stage,
                            "remove dead before type inference for testing")
                self.add_typing_stage(pm)
                self.add_optimization_stage(pm)
                pm.add_stage(self.stage_ir_legalization,
                            "ensure IR is legal prior to lowering")
                self.add_lowering_stage(pm)
                self.add_cleanup_stage(pm)

            def rm_dead_stage(self):
                numba.ir_utils.remove_dead(
                    self.func_ir.blocks, self.func_ir.arg_names, self.func_ir)

        @numba.jit(pipeline_class=TestPipeline)
        def f(a):
            return 0

        with self.assertRaises(errors.TypingError) as raises:
            f(iter([1,2]))  # use a type that Numba doesn't recognize

        expected = 'File "unknown location", line 0:'
        self.assertIn(expected, str(raises.exception))


class TestConstantInferenceErrorHandling(unittest.TestCase):

    def test_basic_error(self):
        # issue 3717
        @njit
        def problem(a,b):
            if a == b:
                raise Exception("Equal numbers: %i %i", a, b)
            return a * b

        with self.assertRaises(errors.ConstantInferenceError) as raises:
            problem(1,2)

        msg1 = "Constant inference not possible for: arg(0, name=a)"
        msg2 = 'raise Exception("Equal numbers: %i %i", a, b)'
        self.assertIn(msg1, str(raises.exception))
        self.assertIn(msg2, str(raises.exception))


if __name__ == '__main__':
    unittest.main()
