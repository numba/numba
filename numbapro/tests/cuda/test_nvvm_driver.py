import unittest, os

from numbapro.cudapipeline.nvvm import *
from ctypes import c_size_t, c_uint64, sizeof
from llvm.core import *
import support

is64bit = sizeof(c_size_t) == sizeof(c_uint64)

class TestNvvmDriver(support.CudaTestCase):

    def get_ptx(self):
        directory = os.path.dirname(__file__)
        nvvm = NVVM()
        print nvvm.get_version()

        if is64bit:
            filename = os.path.join(directory, 'simple-gpu64.ll')
            with open(filename) as fin:
                return fin.read()
        else:
            filename = os.path.join(directory, 'simple-gpu.ll')
            with open(filename) as fin:
                return fin.read()


    def test_nvvm_compile(self):
        nvvmir = self.get_ptx()
        cu = CompilationUnit()

        cu.add_module(nvvmir)
        ptx = cu.compile()
        print ptx
        self.assertTrue('simple' in ptx)
        self.assertTrue('ave' in ptx)
        print cu.log

    def test_nvvm_compile_simple(self):
        nvvmir = self.get_ptx()
        ptx = llvm_to_ptx(nvvmir)
        print ptx
        self.assertTrue('simple' in ptx)
        self.assertTrue('ave' in ptx)

    def test_nvvm_from_llvm(self):
        m = Module.new("test_nvvm_from_llvm")
        fty = Type.function(Type.void(), [Type.int()])
        kernel = m.add_function(fty, name='mycudakernel')
        bldr = Builder.new(kernel.append_basic_block('entry'))
        bldr.ret_void()
        print m
        set_cuda_kernel(kernel)

        fix_data_layout(m)
        ptx = llvm_to_ptx(str(m))
        print ptx
        self.assertTrue('mycudakernel' in ptx)
        if is64bit:
            self.assertTrue('.address_size 64' in ptx)
        else:
            self.assertTrue('.address_size 32' in ptx)

if __name__ == '__main__':
    unittest.main()

