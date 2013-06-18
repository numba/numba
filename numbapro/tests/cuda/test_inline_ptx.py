import unittest
import numpy as np
import math
from numba import *
from numbapro.cudadrv import nvvm
from llvm.core import *
import support
#import logging; logging.getLogger().setLevel(1)

class TestCudaInlineAsm(support.CudaTestCase):
    def test_inline_rsqrt(self):
        mod = Module.new(__name__)
        fnty = Type.function(Type.void(), [Type.pointer(Type.float())])
        fn = mod.add_function(fnty, 'cu_rsqrt')
        bldr = Builder.new(fn.append_basic_block('entry'))

        rsqrt_approx_fnty = Type.function(Type.float(), [Type.float()])
        inlineasm = InlineAsm.get(rsqrt_approx_fnty,
                                  'rsqrt.approx.f32 $0, $1;',
                                  '=f,f', side_effect=True)
        val = bldr.load(fn.args[0])
        res = bldr.call(inlineasm, [val])

        bldr.store(res, fn.args[0])
        bldr.ret_void()

        # generate ptx
        nvvm.fix_data_layout(mod)
        nvvm.set_cuda_kernel(fn)
        nvvmir = str(mod)
        ptx = nvvm.llvm_to_ptx(nvvmir)
        print ptx

        self.assertTrue('rsqrt.approx.f32' in ptx)


if __name__ == '__main__':
    unittest.main()


