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

import unittest
from numba.tests.support import temp_directory, SerialMixin
from numba.core import dispatcher


@contextlib.contextmanager
def captured_logs(l):
    try:
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        l.addHandler(handler)
        yield buffer
    finally:
        l.removeHandler(handler)


class TestJitModule(SerialMixin, unittest.TestCase):

    source_lines = """
from numba import jit_module

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

jit_module({jit_options})
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

    def test_jit_module(self):
        with self.create_temp_jitted_module() as test_module:
            self.assertIsInstance(test_module.inc, dispatcher.Dispatcher)
            self.assertIsInstance(test_module.add, dispatcher.Dispatcher)
            self.assertIsInstance(test_module.inc_add, dispatcher.Dispatcher)
            self.assertTrue(test_module.mean is np.mean)
            self.assertTrue(inspect.isclass(test_module.Foo))

            # Test output of jitted functions is as expected
            x, y = 1.7, 2.3
            self.assertEqual(test_module.inc(x),
                             test_module.inc.py_func(x))
            self.assertEqual(test_module.add(x, y),
                             test_module.add.py_func(x, y))
            self.assertEqual(test_module.inc_add(x),
                             test_module.inc_add.py_func(x))

    def test_jit_module_jit_options(self):
        jit_options = {"nopython": True,
                       "nogil": False,
                       "error_model": "numpy",
                       "boundscheck": False,
                       }
        with self.create_temp_jitted_module(**jit_options) as test_module:
            self.assertEqual(test_module.inc.targetoptions, jit_options)

    def test_jit_module_jit_options_override(self):
        source_lines = """
from numba import jit, jit_module

@jit(nogil=True, forceobj=True)
def inc(x):
    return x + 1

def add(x, y):
    return x + y

jit_module({jit_options})
"""
        jit_options = {"nopython": True,
                       "error_model": "numpy",
                       "boundscheck": False,
                       }
        with self.create_temp_jitted_module(source_lines=source_lines,
                                            **jit_options) as test_module:
            self.assertEqual(test_module.add.targetoptions, jit_options)
            # Test that manual jit-wrapping overrides jit_module options
            self.assertEqual(test_module.inc.targetoptions,
                             {'nogil': True, 'forceobj': True,
                              'boundscheck': None})

    def test_jit_module_logging_output(self):
        logger = logging.getLogger('numba.core.decorators')
        logger.setLevel(logging.DEBUG)
        jit_options = {"nopython": True,
                       "error_model": "numpy",
                       }
        with captured_logs(logger) as logs:
            with self.create_temp_jitted_module(**jit_options) as test_module:
                logs = logs.getvalue()
                expected = ["Auto decorating function",
                            "from module {}".format(test_module.__name__),
                            "with jit and options: {}".format(jit_options)]
                self.assertTrue(all(i in logs for i in expected))

    def test_jit_module_logging_level(self):
        logger = logging.getLogger('numba.core.decorators')
        # Test there's no logging for INFO level
        logger.setLevel(logging.INFO)
        with captured_logs(logger) as logs:
            with self.create_temp_jitted_module():
                self.assertEqual(logs.getvalue(), '')
