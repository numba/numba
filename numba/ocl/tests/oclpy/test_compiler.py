from __future__ import print_function, absolute_import
from numba import unittest_support as unittest
from numba.ocl.ocldrv import cl
from numba.ocl import compiler
from numba import types
import numpy as np
from ctypes import c_size_t, addressof, sizeof

class TestCompiler(unittest.TestCase):
    def test_compiler_small(self):
        def pyfunc(x):
            x[0] = 1234

        argtys = [types.Array(types.int32, 1, 'C')]
        cres = compiler.compile_kernel(pyfunc, argtys)
        print(cres.llvm_module)

        # Try loading it
        bc = cres.llvm_module.to_bitcode()
        device = cl.default_platform.default_device
        context = cl.create_context(device.platform, [device])
        program = context.create_program_from_binary(bc)
        program.build(options=b"-x spir -spir-std=1.2")
        print(program.kernel_names)
        kernel = program.create_kernel(program.kernel_names[0].encode('utf8'))

        self.assertEqual(kernel.num_args, 3)

        # Call
        result = np.zeros(1, dtype='int32')
        q = context.create_command_queue(device)
        buf = context.create_buffer(4)
        kernel.set_arg(0, buf)

        shape = c_size_t(1)
        stride = c_size_t(4)
        kernel.set_arg_raw(1, addressof(shape), sizeof(shape))   # shape
        kernel.set_arg_raw(2, addressof(stride), sizeof(stride))   # strides

        local_sz = 1
        global_sz = 1
        q.enqueue_nd_range_kernel(kernel, 1, [global_sz], [local_sz])
        q.finish()
        q.enqueue_read_buffer(buf, 0, 4, result.ctypes.data)

        print(result)
        self.assertEqual(result[0], 1234)

if __name__ == '__main__':
    unittest.main()

