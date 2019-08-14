from __future__ import print_function

import builtins
import types as pytypes

import numpy as np

from numba import types
from .support import TestCase
from numba.help.inspector import inspect_function, inspect_module


class TestInspector(TestCase):
    def check_function_descriptor(self, info, must_be_defined=False):
        self.assertIsInstance(info, dict)
        self.assertIn('numba_type', info)
        numba_type = info['numba_type']
        if numba_type is None:
            self.assertFalse(must_be_defined)
        else:
            self.assertIsInstance(numba_type, types.Type)
            self.assertIn('explained', info)
            self.assertIsInstance(info['explained'], str)
            self.assertIn('source_infos', info)
            self.assertIsInstance(info['source_infos'], dict)

    def test_inspect_function_on_range(self):
        info = inspect_function(range)
        self.check_function_descriptor(info, must_be_defined=True)

    def test_inspect_function_on_np_all(self):
        info = inspect_function(np.all)
        self.check_function_descriptor(info, must_be_defined=True)
        source_infos = info['source_infos']
        self.assertGreater(len(source_infos), 0)
        c = 0
        for srcinfo in source_infos.values():
            self.assertIsInstance(srcinfo['kind'], str)
            self.assertIsInstance(srcinfo['name'], str)
            self.assertIsInstance(srcinfo['sig'], str)
            self.assertIsInstance(srcinfo['filename'], str)
            self.assertIsInstance(srcinfo['lines'], tuple)
            self.assertIn('docstring', srcinfo)
            c += 1
        self.assertEqual(c, len(source_infos))

    def test_inspect_module(self):
        c = 0
        for it in inspect_module(builtins):
            self.assertIsInstance(it['module'], pytypes.ModuleType)
            self.assertIsInstance(it['name'], str)
            self.assertTrue(callable(it['obj']))
            self.check_function_descriptor(it)
            c += 1
        self.assertGreater(c, 0)
