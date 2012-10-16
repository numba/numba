import unittest

from numbapro._cuda.nvvm import *
from ctypes import c_size_t, c_uint64, sizeof

class TestNvvmDriver(unittest.TestCase):

    def get_ptx(self):
        is64bit = sizeof(c_size_t) == sizeof(c_uint64)

        nvvm = NVVM()
        print nvvm.get_version()

        if is64bit:
            with open('simple-gpu64.ll') as fin:
                return fin.read()
        else:
            with open('simple-gpu.ll') as fin:
                return fin.read()


    def test_nvvm_compile(self):
        nvvmir = self.get_ptx()
        cu = CompilationUnit()

        cu.add_module(nvvmir)
        ptx = cu.compile()
        print ptx
        self.assertIn('simple', ptx)
        self.assertIn('ave', ptx)
        print cu.log

    def test_nvvm_compile_simple(self):
        nvvmir = self.get_ptx()
        ptx = llvm_to_ptx(nvvmir)
        print ptx
        self.assertIn('simple', ptx)
        self.assertIn('ave', ptx)

if __name__ == '__main__':
    unittest.main()

