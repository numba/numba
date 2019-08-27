
import os
import sys
import shutil
import inspect
import importlib
import contextlib
import uuid
import numpy as np
import logging

import numba.unittest_support as unittest
from numba import dispatcher
from numba.utils import StringIO
from numba.tests.support import temp_directory, SerialMixin

logger = logging.getLogger('numba.decorators')


class CapturedLog:
    """Capture the log temporarily for validation."""

    def __init__(self):
        self.buffer = StringIO()
        self.handler = logging.StreamHandler(self.buffer)

    def __enter__(self):
        self._handlers = logger.handlers
        self.buffer = StringIO()
        logger.handlers = [logging.StreamHandler(self.buffer)]

    def __exit__(self, type, value, traceback):
        logger.handlers = self._handlers

    def getvalue(self):
        return self.buffer.getvalue()


class TestJitModule(SerialMixin, unittest.TestCase):

    def setUp(self):
        self.capture = CapturedLog()

    def tearDown(self):
        del self.capture

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
        if source_lines is None:
            source_lines = self.source_lines
        tempdir = temp_directory('test_jit_module')
        # Generate random module name
        temp_module_name = 'test_module_' + str(uuid.uuid4()).replace('-', '_')
        temp_module_path = os.path.join(tempdir, temp_module_name + '.py')

        jit_options = self._format_jit_options(**jit_options)
        with open(temp_module_path, 'w') as f:
            lines = source_lines.format(jit_options=jit_options)
            f.write(lines)
        # Add test_module to sys.path so it can be imported
        sys.path.insert(0, tempdir)
        test_module = importlib.import_module(temp_module_name)
        yield test_module
        sys.modules.pop(temp_module_name, None)
        sys.path.remove(tempdir)
        shutil.rmtree(tempdir)

    def test_create_temp_jitted_module(self):
        sys_path_original = list(sys.path)
        with self.create_temp_jitted_module() as test_module:
            temp_module_dir = os.path.dirname(test_module.__file__)
            self.assertTrue(temp_module_dir == sys.path[0])
            self.assertTrue(sys.path[1:] == sys_path_original)
        # Make sure modifications to sys.path are reverted by context manager
        self.assertTrue(sys.path == sys_path_original)

    def test_jit_module(self):
        with self.create_temp_jitted_module() as test_module:
            self.assertTrue(isinstance(test_module.inc, dispatcher.Dispatcher))
            self.assertTrue(isinstance(test_module.add, dispatcher.Dispatcher))
            self.assertTrue(isinstance(test_module.inc_add, dispatcher.Dispatcher))
            self.assertTrue(test_module.mean is np.mean)
            self.assertTrue(inspect.isclass(test_module.Foo))

            # Test output of jitted functions is as expected
            x, y = 1.7, 2.3
            self.assertEqual(test_module.inc(x), test_module.inc.py_func(x))
            self.assertEqual(test_module.add(x, y), test_module.add.py_func(x, y))
            self.assertEqual(test_module.inc_add(x), test_module.inc_add.py_func(x))

    def test_jit_module_jit_options(self):
        jit_options = {"nopython": True,
                       "nogil": False,
                       "error_model": "numpy",
                       }
        with self.create_temp_jitted_module(**jit_options) as test_module:
            self.assertTrue(test_module.inc.targetoptions == jit_options)

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
                       }
        with self.create_temp_jitted_module(source_lines=source_lines, **jit_options) as test_module:
            self.assertTrue(test_module.add.targetoptions == jit_options)
            # Test that manual jit-wrapping overrides jit_module options
            self.assertTrue(test_module.inc.targetoptions == {'nogil': True, 'forceobj': True})

    def test_jit_module_logging(self):
        logger = logging.getLogger('numba.decorators')
        logger.setLevel(logging.DEBUG)
        jit_options = {"nopython": True,
                       "error_model": "numpy",
                       }
        with self.capture:
            with self.create_temp_jitted_module(**jit_options) as test_module:
                logs = self.capture.getvalue()
                expected = ["Auto decorating function",
                            "from module {}".format(test_module.__name__),
                            "with jit and options: {}".format(jit_options)]
                self.assertTrue(all(i in logs for i in expected))
