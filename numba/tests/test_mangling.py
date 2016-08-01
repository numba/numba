# -*- coding: utf-8 -*-
"""
Test function name mangling.
The mangling affects the ABI of numba compiled binaries.
"""

from numba import types
from numba.funcdesc import default_mangler
from .support import unittest, TestCase


class TestMangling(TestCase):
    def test_one_args(self):
        fname = 'foo'
        argtypes = types.int32,
        name = default_mangler(fname, argtypes)
        self.assertEqual(name, 'foo.int32')

    def test_two_args(self):
        fname = 'foo'
        argtypes = types.int32, types.float32
        name = default_mangler(fname, argtypes)
        self.assertEqual(name, 'foo.int32.float32')

    def test_unicode_fname(self):
        fname = u'foಠ'
        argtypes = types.int32, types.float32
        name = default_mangler(fname, argtypes)
        self.assertIsInstance(name, str)
        expect = 'fo\\u{0:04x}.int32.float32'.format(ord(u'ಠ'))
        self.assertEqual(name, expect)


if __name__ == '__main__':
    unittest.main()
