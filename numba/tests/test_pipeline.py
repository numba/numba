from __future__ import print_function

from numba.compiler import Pipeline
from numba import jit, generated_jit, types
from .support import TestCase


class TestCustomPipeline(TestCase):
    def setUp(self):
        super(TestCustomPipeline, self).setUp()

        # Define custom pipeline class
        class CustomPipeline(Pipeline):
            custom_pipeline_cache = []

            def compile_extra(self, func):
                # Store the compiled function
                self.custom_pipeline_cache.append(func)
                return super(CustomPipeline, self).compile_extra(func)

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
