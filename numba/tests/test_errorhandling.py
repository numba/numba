"""
Unspecified error handling tests
"""

from numba import jit, njit, typed, int64
from numba.core import errors, utils
import numpy as np


from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
                                       IRProcessing,)

from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
                                     NativeLowering, IRLegalization,
                                     NoPythonBackend)

from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass

from numba.tests.support import skip_parfors_unsupported
import unittest

# used in TestMiscErrorHandling::test_handling_of_write_to_*_global
_global_list = [1, 2, 3, 4]
_global_dict = typed.Dict.empty(int64, int64)

class TestErrorHandlingBeforeLowering(unittest.TestCase):

    def test_unsupported_make_function_return_inner_func(self):
        def func(x):
            """ return the closure """
            z = x + 1

            def inner(x):
                return x + z
            return inner

        for pipeline in jit, njit:
            with self.assertRaises(errors.TypingError) as raises:
                pipeline(func)(1)

            expected = "Cannot capture the non-constant value"
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

    def test_commented_func_definition_is_not_a_definition(self):
        # See issue #4056, the commented def should not be found as the
        # definition for reporting purposes when creating the synthetic
        # traceback because it is commented! Use of def in docstring would also
        # cause this issue hence is tested.

        def foo_commented():
            #def commented_definition()
            raise Exception('test_string')

        def foo_docstring():
            """ def docstring containing def might match function definition!"""
            raise Exception('test_string')

        for func in (foo_commented, foo_docstring):
            with self.assertRaises(Exception) as raises:
                func()

            self.assertIn("test_string", str(raises.exception))

    def test_use_of_ir_unknown_loc(self):
        # for context see # 3390
        from numba.core.compiler import CompilerBase
        class TestPipeline(CompilerBase):
            def define_pipelines(self):
                name = 'bad_DCE_pipeline'
                pm = PassManager(name)
                pm.add_pass(TranslateByteCode, "analyzing bytecode")
                pm.add_pass(FixupArgs, "fix up args")
                pm.add_pass(IRProcessing, "processing IR")
                # remove dead before type inference so that the Arg node is removed
                # and the location of the arg cannot be found
                pm.add_pass(DeadCodeElimination, "DCE")
                # typing
                pm.add_pass(NopythonTypeInference, "nopython frontend")
                pm.add_pass(NoPythonBackend, "nopython mode backend")
                pm.finalize()
                return [pm]

        @njit(pipeline_class=TestPipeline)
        def f(a):
            return 0

        with self.assertRaises(errors.TypingError) as raises:
            f(iter([1,2]))  # use a type that Numba doesn't recognize

        expected = 'File "unknown location", line 0:'
        self.assertIn(expected, str(raises.exception))

    def check_write_to_globals(self, func):
        with self.assertRaises(errors.TypingError) as raises:
            func()

        expected = ["The use of a", "in globals, is not supported as globals"]
        for ex in expected:
            self.assertIn(ex, str(raises.exception))


    def test_handling_of_write_to_reflected_global(self):
        @njit
        def foo():
            _global_list[0] = 10

        self.check_write_to_globals(foo)

    def test_handling_of_write_to_typed_dict_global(self):
        @njit
        def foo():
            _global_dict[0] = 10

        self.check_write_to_globals(foo)

    @skip_parfors_unsupported
    def test_handling_forgotten_numba_internal_import(self):
        @njit(parallel=True)
        def foo():
            for i in prange(10): # prange is not imported
                pass

        with self.assertRaises(errors.TypingError) as raises:
            foo()

        expected = ("'prange' looks like a Numba internal function, "
                    "has it been imported")
        self.assertIn(expected, str(raises.exception))

    def test_handling_unsupported_generator_expression(self):
        def foo():
            y = (x for x in range(10))

        expected = "The use of yield in a closure is unsupported."

        for dec in jit(forceobj=True), njit:
            with self.assertRaises(errors.UnsupportedError) as raises:
                dec(foo)()
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
