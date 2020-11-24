import os
import inspect
import numpy as np
import logging

from numba import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.test_jit_module import BaseJitModuleTest, captured_logs
from numba.cuda.compiler import DeviceFunctionTemplate, Dispatcher
from numba.cuda.simulator.kernel import FakeCUDAKernel


class TestJitModule(BaseJitModuleTest, CUDATestCase):

    source_lines = """
from numba import cuda

def inc(x):
    return x + 1

def add(x, y):
    return x + y

def inc_add(x):
    y = inc(x)
    return add(x, y)

import numpy as np
mean = np.mean

class Foo(object):
    pass

cuda.jit_module({jit_options})
"""

    def test_jit_module_device(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        source_path = os.path.join(test_dir, 'jit_module_device.py')
        with open(source_path) as f:
            source = f.read()

        with self.create_temp_jitted_module(source_lines=source) as test_module:
            if config.ENABLE_CUDASIM:
                self.assertIsInstance(test_module.inc, FakeCUDAKernel)
                self.assertTrue(test_module.inc._device)
                self.assertIsInstance(test_module.add, FakeCUDAKernel)
                self.assertTrue(test_module.add._device)
                self.assertIsInstance(test_module.inc_add, FakeCUDAKernel)
                self.assertTrue(test_module.inc_add._device)
            else:
                self.assertIsInstance(test_module.inc, DeviceFunctionTemplate)
                self.assertIsInstance(test_module.add, DeviceFunctionTemplate)
                self.assertIsInstance(test_module.inc_add,
                                      DeviceFunctionTemplate)

            self.assertTrue(test_module.mean is np.mean)
            self.assertTrue(inspect.isclass(test_module.Foo))

            # Test output of jitted functions is as expected
            x, y = 1.7, 2.3
            self.assertEqual(test_module.Callers.inc(x),
                             test_module.inc.py_func(x))
            self.assertEqual(test_module.Callers.add(x, y),
                             test_module.add.py_func(x, y))

            # We can't call inc_add directly for testing as a Python function,
            # because it calls inc and add - these are DeviceFunctionTemplates,
            # not Python functions. We manually construct the expected result
            # using inc.py_func and add.py_func instead.
            y = test_module.inc.py_func(x)
            add_x = test_module.add.py_func(x, y)
            self.assertEqual(test_module.Callers.inc_add(x),
                             add_x)

    def test_jit_module_kernel(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        source_path = os.path.join(test_dir, 'jit_module_kernel.py')
        with open(source_path) as f:
            source = f.read()

        with self.create_temp_jitted_module(source_lines=source) as test_module:
            if config.ENABLE_CUDASIM:
                self.assertIsInstance(test_module.inc, FakeCUDAKernel)
                self.assertFalse(test_module.inc._device)
                self.assertIsInstance(test_module.add, FakeCUDAKernel)
                self.assertFalse(test_module.add._device)
                self.assertIsInstance(test_module.inc_add, FakeCUDAKernel)
                self.assertFalse(test_module.inc_add._device)
            else:
                self.assertIsInstance(test_module.inc, Dispatcher)
                self.assertIsInstance(test_module.add, Dispatcher)
                self.assertIsInstance(test_module.inc_add, Dispatcher)

            self.assertTrue(test_module.mean is np.mean)
            self.assertTrue(inspect.isclass(test_module.Foo))

            # Test output of jitted functions is as expected. Here we use some
            # helper functions from the module for calling the Python versions
            # of the functions as well as the jitted ones.
            x, y = 1.7, 2.3
            self.assertEqual(test_module.Callers.inc(x),
                             test_module.Callers.py_inc(x))
            self.assertEqual(test_module.Callers.add(x, y),
                             test_module.Callers.py_add(x, y))
            self.assertEqual(test_module.Callers.inc_add(x),
                             test_module.Callers.py_inc_add(x))

    def test_jit_module_both(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        source_path = os.path.join(test_dir, 'jit_module_both.py')
        with open(source_path) as f:
            source = f.read()

        with self.create_temp_jitted_module(source_lines=source) as test_module:
            if config.ENABLE_CUDASIM:
                self.assertIsInstance(test_module.inc, FakeCUDAKernel)
                self.assertTrue(test_module.inc._device)
                self.assertIsInstance(test_module.add, FakeCUDAKernel)
                self.assertTrue(test_module.add._device)
                self.assertIsInstance(test_module.inc_add, FakeCUDAKernel)
                self.assertFalse(test_module.inc_add._device)
            else:
                self.assertIsInstance(test_module.inc, DeviceFunctionTemplate)
                self.assertIsInstance(test_module.add, DeviceFunctionTemplate)
                self.assertIsInstance(test_module.inc_add, Dispatcher)

            self.assertTrue(test_module.mean is np.mean)
            self.assertTrue(inspect.isclass(test_module.Foo))

            # Test output of jitted functions is as expected. Here we use some
            # helper functions from the module for calling the Python versions
            # of the functions as well as the jitted ones.
            x, y = 1.7, 2.3
            self.assertEqual(test_module.call_inc(x),
                             test_module.inc.py_func(x))
            self.assertEqual(test_module.call_add(x, y),
                             test_module.add.py_func(x, y))
            self.assertEqual(test_module.call_inc_add(x),
                             test_module.call_py_inc_add(x))

    @skip_on_cudasim('options ignored by cudasim')
    def test_jit_module_jit_options(self):
        jit_options = { "debug": True }
        with self.create_temp_jitted_module(**jit_options) as test_module:
            self.assertTrue(test_module.inc.debug)

    @skip_on_cudasim('options ignored by cudasim')
    def test_jit_module_jit_options_override(self):
        source_lines = """
from numba import cuda

@cuda.jit(debug=True)
def inc(x):
    return x + 1

def add(x, y):
    return x + y

cuda.jit_module({jit_options})
"""
        jit_options = {"debug": False,
                       "opt": False,
                       }
        with self.create_temp_jitted_module(source_lines=source_lines,
                                            **jit_options) as test_module:
            self.assertEqual(test_module.add.debug, False)
            self.assertEqual(test_module.add.opt, False)
            # Test that manual jit-wrapping overrides jit_module options
            inc_opts = test_module.inc.targetoptions
            self.assertEqual(inc_opts['debug'], True)
            self.assertEqual(inc_opts['opt'], True)

    def test_jit_module_logging_output(self):
        logger = logging.getLogger('numba.cuda.decorators')
        logger.setLevel(logging.DEBUG)
        jit_options = {"debug": True,
                       "inline": False,
                       "device": True
                       }
        with captured_logs(logger) as logs:
            with self.create_temp_jitted_module(**jit_options) as test_module:
                logs = logs.getvalue()
                expected = ["Auto decorating function",
                            "from module {}".format(test_module.__name__),
                            "with jit and options: {}".format(jit_options)]
                for i in expected:
                    self.assertIn(i, logs)

    def test_jit_module_logging_level(self):
        logger = logging.getLogger('numba.cuda.decorators')
        # Test there's no logging for INFO level
        logger.setLevel(logging.INFO)
        with captured_logs(logger) as logs:
            with self.create_temp_jitted_module():
                self.assertEqual(logs.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
