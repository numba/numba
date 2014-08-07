"""
Test helper functions from numba.numpy_support.
"""

from __future__ import print_function

import numpy as np

import numba.unittest_support as unittest
from numba import config, numpy_support, types
from .support import TestCase


class TestFromDtype(TestCase):

    def test_number_types(self):
        """
        Test the various scalar number types.
        """
        f = numpy_support.from_dtype

        def check(typechar, numba_type):
            # Only native ordering and alignment is supported
            self.assertIs(f(np.dtype(typechar)), numba_type)
            self.assertIs(f(np.dtype('=' + typechar)), numba_type)

        check('?', types.bool_)
        check('f', types.float32)
        check('f4', types.float32)
        check('d', types.float64)
        check('f8', types.float64)

        check('F', types.complex64)
        check('c8', types.complex64)
        check('D', types.complex128)
        check('c16', types.complex128)

        check('b', types.int8)
        check('i1', types.int8)
        check('B', types.uint8)
        check('u1', types.uint8)

        check('h', types.int16)
        check('i2', types.int16)
        check('H', types.uint16)
        check('u2', types.uint16)

        check('i', types.int32)
        check('i4', types.int32)
        check('I', types.uint32)
        check('u4', types.uint32)

        check('q', types.int64)
        check('Q', types.uint64)
        for name in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
                     'int64', 'uint64', 'intp', 'uintp'):
            self.assertIs(f(np.dtype(name)), getattr(types, name))

        # Non-native alignments are unsupported (except for 1-byte types)
        foreign_align = '>' if config.NATIVE_ALIGNMENT == 'little' else '<'
        for letter in 'hHiIlLqQfdFD':
            self.assertRaises(NotImplementedError, f,
                              np.dtype(foreign_align + letter))

    def test_string_types(self):
        """
        Test the character string types.
        """
        f = numpy_support.from_dtype
        self.assertEqual(f(np.dtype('S10')), types.CharSeq(10))
        self.assertEqual(f(np.dtype('a11')), types.CharSeq(11))
        self.assertEqual(f(np.dtype('U12')), types.UnicodeCharSeq(12))

    def test_timedelta_types(self):
        """
        Test the timedelta types.
        """
        f = numpy_support.from_dtype
        tp = f(np.dtype('m'))
        self.assertEqual(tp, types.NPTimedelta(''))
        for code, unit in enumerate(('Y', 'M', 'W')):
            tp = f(np.dtype('m8[%s]' % unit))
            self.assertEqual(tp, types.NPTimedelta(unit))
            self.assertEqual(tp.unit_code, code)
        for code, unit in enumerate(
            ('D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'),
            start=4):
            tp = f(np.dtype('m8[%s]' % unit))
            self.assertEqual(tp, types.NPTimedelta(unit))
            self.assertEqual(tp.unit_code, code)


if __name__ == '__main__':
    unittest.main()

