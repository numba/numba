from __future__ import print_function, division, absolute_import

from llvmlite.llvmpy.core import Module, Type, Builder, InlineAsm
from llvmlite import binding as ll

from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('Inline PTX cannot be used in the simulator')
class TestCudaInlineAsm(CUDATestCase):
    def test_inline_rsqrt(self):
        mod = Module(__name__)
        fnty = Type.function(Type.void(), [Type.pointer(Type.float())])
        fn = mod.add_function(fnty, 'cu_rsqrt')
        bldr = Builder(fn.append_basic_block('entry'))

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
        self.assertTrue('rsqrt.approx.f32' in str(ptx))


if __name__ == '__main__':
    unittest.main()
