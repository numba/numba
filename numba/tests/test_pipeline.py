from __future__ import print_function

from numba.compiler import Compiler
from numba import jit, generated_jit, types, objmode
from numba.ir import FunctionIR
from .support import TestCase


class TestCustomPipeline(TestCase):
    def setUp(self):
        super(TestCustomPipeline, self).setUp()

        # Define custom pipeline class
        class CustomPipeline(Compiler):
            custom_pipeline_cache = []

            def compile_extra(self, func):
                # Store the compiled function
                self.custom_pipeline_cache.append(func)
                return super(CustomPipeline, self).compile_extra(func)

            def compile_ir(self, func_ir, *args, **kwargs):
                # Store the compiled function
                self.custom_pipeline_cache.append(func_ir)
                return super(CustomPipeline, self).compile_ir(
                    func_ir, *args, **kwargs)

        self.pipeline_class = CustomPipeline

    def test_jit_custom_pipeline(self):
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [])

        @jit(pipeline_class=self.pipeline_class)
        def foo(x):
            return x

        self.assertEqual(foo(4), 4)
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache,
                             [foo.py_func])

    def test_generated_jit_custom_pipeline(self):
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [])

        def inner(x):
            return x

        @generated_jit(pipeline_class=self.pipeline_class)
        def foo(x):
            if isinstance(x, types.Integer):
                return inner

        self.assertEqual(foo(5), 5)
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache,
                             [inner])

    def test_objmode_custom_pipeline(self):
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [])

        @jit(pipeline_class=self.pipeline_class)
        def foo(x):
            with objmode(x="intp"):
                x += int(0x1)
            return x

        arg = 123
        self.assertEqual(foo(arg), arg + 1)
        # Two items in the list.
        self.assertEqual(len(self.pipeline_class.custom_pipeline_cache), 2)
        # First item is the `foo` function
        first = self.pipeline_class.custom_pipeline_cache[0]
        self.assertIs(first, foo.py_func)
        # Second item is a FunctionIR of the obj-lifted function
        second = self.pipeline_class.custom_pipeline_cache[1]
        self.assertIsInstance(second, FunctionIR)

