import os
import sys
import shutil
import inspect
import importlib
import contextlib
import uuid
import numpy as np
import logging
from io import StringIO

from numba.cuda.testing import unittest, CUDATestCase
from numba.tests.support import temp_directory
from numba.cuda.compiler import DeviceFunctionTemplate, Dispatcher


@contextlib.contextmanager
def captured_logs(l):
    try:
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        l.addHandler(handler)
        yield buffer
    finally:
        l.removeHandler(handler)


class TestJitModule(CUDATestCase):

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

    def _format_jit_options(self, **jit_options):
        if not jit_options:
            return ''
        out = []
        for key, value in jit_options.items():
            if isinstance(value, str):
                value = '"{}"'.format(value)
            out.append('{}={}'.format(key, value))
        return ', '.join(out)

    @contextlib.contextmanager
    def create_temp_jitted_module(self, source_lines=None, **jit_options):
        # Use try/finally so cleanup happens even when an exception is raised
        try:
            if source_lines is None:
                source_lines = self.source_lines
            tempdir = temp_directory('test_jit_module')
            # Generate random module name
            temp_module_name = 'test_module_{}'.format(
                str(uuid.uuid4()).replace('-', '_'))
            temp_module_path = os.path.join(tempdir, temp_module_name + '.py')

            jit_options = self._format_jit_options(**jit_options)
            with open(temp_module_path, 'w') as f:
                lines = source_lines.format(jit_options=jit_options)
                f.write(lines)
            # Add test_module to sys.path so it can be imported
            sys.path.insert(0, tempdir)
            test_module = importlib.import_module(temp_module_name)
            yield test_module
        finally:
            sys.modules.pop(temp_module_name, None)
            sys.path.remove(tempdir)
            shutil.rmtree(tempdir)

    def test_create_temp_jitted_module(self):
        sys_path_original = list(sys.path)
        sys_modules_original = dict(sys.modules)
        with self.create_temp_jitted_module() as test_module:
            temp_module_dir = os.path.dirname(test_module.__file__)
            self.assertEqual(temp_module_dir, sys.path[0])
            self.assertEqual(sys.path[1:], sys_path_original)
            self.assertTrue(test_module.__name__ in sys.modules)
        # Test that modifications to sys.path / sys.modules are reverted
        self.assertEqual(sys.path, sys_path_original)
        self.assertEqual(sys.modules, sys_modules_original)

    def test_create_temp_jitted_module_with_exception(self):
        try:
            sys_path_original = list(sys.path)
            sys_modules_original = dict(sys.modules)
            with self.create_temp_jitted_module():
                raise ValueError("Something went wrong!")
        except ValueError:
            # Test that modifications to sys.path / sys.modules are reverted
            self.assertEqual(sys.path, sys_path_original)
            self.assertEqual(sys.modules, sys_modules_original)

    def test_jit_module_device(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        source_path = os.path.join(test_dir, 'jit_module_device.py')
        with open(source_path) as f:
            source = f.read()

        with self.create_temp_jitted_module(source_lines=source) as test_module:
            self.assertIsInstance(test_module.inc, DeviceFunctionTemplate)
            self.assertIsInstance(test_module.add, DeviceFunctionTemplate)
            self.assertIsInstance(test_module.inc_add, DeviceFunctionTemplate)
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

    def test_jit_module_jit_options(self):
        jit_options = { "debug": True }
        with self.create_temp_jitted_module(**jit_options) as test_module:
            self.assertTrue(test_module.inc.debug)

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
