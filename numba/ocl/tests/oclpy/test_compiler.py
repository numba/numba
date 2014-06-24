from __future__ import print_function, absolute_import
from numba import unittest_support as unittest
from numba.ocl.ocldrv import cl
from numba.ocl import compiler
from numba import types


class TestCompiler(unittest.TestCase):
    def test_compiler_small(self):
        def pyfunc(x):
            x[0] = 1234

        argtys = [types.Array(types.int32, 1, 'A')]
        cres = compiler.compile_kernel(pyfunc, argtys)
        print(cres.llvm_module)

        # Try loading it
        bc = cres.llvm_module.to_bitcode()
        device = cl.default_platform.default_device
        context = cl.create_context(device.platform, [device])
        program = context.create_program_from_binary(bc)
        program.build(options=b"-x spir -spir-std=1.2")
        print(program.kernel_names)


if __name__ == '__main__':
    unittest.main()

