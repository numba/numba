import unittest
from numba import types, typelattice


class TestTypeDist(unittest.TestCase):
    def test_int_to_complex(self):
        lattice = typelattice.type_lattice
        self.assertEqual(3.25, lattice[(types.int32, types.complex64)])
        self.assertEqual(2.25, lattice[(types.int64, types.complex64)])
        self.assertEqual(4.25, lattice[(types.int32, types.complex128)])
        self.assertEqual(3.25, lattice[(types.int64, types.complex128)])


if __name__ == '__main__':
    unittest.main()
