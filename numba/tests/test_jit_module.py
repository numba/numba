
import os
import sys
import shutil
import inspect
import importlib
import contextlib
import uuid

import numba.unittest_support as unittest
from numba import jit_module, dispatcher
from numba.tests.support import temp_directory


class TestJitModule(unittest.TestCase):

    source_text_file = """
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

jit_module({name}, {jit_options})
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
    def create_temp_jitted_module(self, module_name, **jit_options):
        tempdir = temp_directory('test_jit_module')
        # Generate random module name
        temp_module_name = 'test_module_' + str(uuid.uuid4()).replace('-', '_')
        temp_module_path = os.path.join(tempdir, temp_module_name + '.py')

        jit_options = self._format_jit_options(**jit_options)
        with open(temp_module_path, 'w') as f:
            lines = self.source_text_file.format(name=module_name,
                                                 jit_options=jit_options)
            f.write(lines)
        # Add test_module to sys.path so it can be imported
        sys.path.insert(0, tempdir)
        test_module = importlib.import_module(temp_module_name)
        yield test_module
        sys.modules.pop(temp_module_name, None)
        sys.path.remove(tempdir)
        shutil.rmtree(tempdir)

    def test_jit_module(self):
        with self.create_temp_jitted_module('__name__') as test_module:
            self.assertTrue(isinstance(test_module.inc, dispatcher.Dispatcher))
            self.assertTrue(isinstance(test_module.add, dispatcher.Dispatcher))
            self.assertTrue(isinstance(test_module.inc_add, dispatcher.Dispatcher))
            self.assertFalse(isinstance(test_module.mean, dispatcher.Dispatcher))
            self.assertTrue(inspect.isclass(test_module.Foo))

    def test_jit_module_jit_options(self):
        jit_options = {"nopython": True,
                       "nogil": False,
                       "error_model": "numpy",
                       }
        with self.create_temp_jitted_module('__name__', **jit_options) as test_module:
            self.assertTrue(test_module.inc.targetoptions == jit_options)

    def test_jit_module_raises_nonexistent_module(self):
        with self.assertRaises(ImportError):
            jit_module('this_module_should_not_exit')
