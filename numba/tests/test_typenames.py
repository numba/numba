from numba.core import types
import unittest


class TestTypeNames(unittest.TestCase):
    def test_numpy_integers(self):
        self.assertEqual(types.int_, types.np_intp)
        self.assertEqual(types.uint, types.np_uintp)


if __name__ == '__main__':
    unittest.main()
