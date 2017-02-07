from __future__ import print_function, absolute_import, division

from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.cudadrv.nvvm import llvm39_to_34_ir


@skip_on_cudasim('Linking unsupported in the simulator')
class TestIRPatch(unittest.TestCase):
    def test_load_rewrite(self):
        text = "%myload = not really"
        out = llvm39_to_34_ir(text)
        # No rewrite
        self.assertEqual(text, out)

        text = "%myload = load i32, i32* val"
        out = llvm39_to_34_ir(text)
        # Rewritten
        self.assertEqual("%myload = load i32* val", out)


if __name__ == '__main__':
    unittest.main()
