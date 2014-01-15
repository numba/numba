import numba.unittest_support as unittest
from numba import types, typelattice


class TestTypeDist(unittest.TestCase):
    def test_int_to_complex(self):
        lattice = typelattice.type_lattice
        self.assertTrue((types.int32, types.complex64) in lattice)
        self.assertTrue((types.int64, types.complex64) in lattice)
        self.assertTrue((types.int32, types.complex128) in lattice)
        self.assertTrue((types.int64, types.complex128) in lattice)

        self.assertFalse((types.complex64, types.int32) in lattice)
        self.assertFalse((types.complex128, types.int32) in lattice)


if __name__ == '__main__':
    unittest.main()
