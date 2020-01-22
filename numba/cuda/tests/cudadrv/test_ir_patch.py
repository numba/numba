from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('Linking unsupported in the simulator')
class TestIRPatch(unittest.TestCase):
    def patch(self, ir):
        # Import here to avoid error in CUDASIM
        from numba.cuda.cudadrv.nvvm import llvm39_to_34_ir

        return llvm39_to_34_ir(ir)

    def test_load_rewrite(self):
        text = "%myload = not really"
        out = self.patch(text)
        # No rewrite
        self.assertEqual(text, out)

        text = "%myload = load i32, i32* val"
        out = self.patch(text)
        # Rewritten
        self.assertEqual("%myload = load i32* val", out)


if __name__ == '__main__':
    unittest.main()
