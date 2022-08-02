from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba import njit
import unittest


def addbytes_len_usecase(a, b):
    output = len(b''.join([a, b]))
    return output


BYTES_EXAMPLES = [
    b"abcdef",
    b"12345",
    b"AAAA",
    b"zzzzz",
    b'z',
    b'1',
    bytes(5),
    bytes(10)
]


class BaseTest(MemoryLeakMixin, TestCase):
    def setUp(self):
        super(BaseTest, self).setUp()


class TestBytesData(BaseTest):
    def test_addbytes_len(self, flags=no_pyobj_flags):
        pyfunc = addbytes_len_usecase
        cfunc = njit(pyfunc)
        num_objs = len(BYTES_EXAMPLES)

        for i in range(num_objs):
            obj1 = BYTES_EXAMPLES[i]
            obj2 = BYTES_EXAMPLES[num_objs - i - 1]
            self.assertEqual(pyfunc(obj1, obj2), cfunc(obj1, obj2))


if __name__ == '__main__':
    unittest.main()
