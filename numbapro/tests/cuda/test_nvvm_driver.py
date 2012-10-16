import unittest

from numbapro._cuda.nvvm import *
from ctypes import c_size_t, c_uint64, sizeof

class TestNvvmDriver(unittest.TestCase):

    def test_nvvm_compile(self):
        is64bit = sizeof(c_size_t) == sizeof(c_uint64)

        nvvm = NVVM()
        print nvvm.get_version()

        cu = CompilationUnit()

        if is64bit:
            with open('simple-gpu64.ll') as fin:
                nvvmir = fin.read()
        else:
            with open('simple-gpu.ll') as fin:
                nvvmir = fin.read()

        cu.add_module(nvvmir)
        ptx = cu.compile()
        print ptx
        self.assertIn('simple', ptx)
        self.assertIn('ave', ptx)
        print cu.log

if __name__ == '__main__':
    unittest.main()

