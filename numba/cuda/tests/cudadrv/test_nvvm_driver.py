from llvmlite.llvmpy.core import Module, Type, Builder
from numba.cuda.cudadrv.nvvm import (llvm_to_ptx, set_cuda_kernel,
                                     fix_data_layout, get_arch_option,
                                     get_supported_ccs)
from ctypes import c_size_t, c_uint64, sizeof
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError
from numba.cuda.testing import skip_on_cudasim

is64bit = sizeof(c_size_t) == sizeof(c_uint64)


@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestNvvmDriver(unittest.TestCase):
    def get_ptx(self):
        if is64bit:
            return gpu64
        else:
            return gpu32

    def test_nvvm_compile_simple(self):
        nvvmir = self.get_ptx()
        ptx = llvm_to_ptx(nvvmir).decode('utf8')
        self.assertTrue('simple' in ptx)
        self.assertTrue('ave' in ptx)

    def test_nvvm_from_llvm(self):
        m = Module("test_nvvm_from_llvm")
        fty = Type.function(Type.void(), [Type.int()])
        kernel = m.add_function(fty, name='mycudakernel')
        bldr = Builder(kernel.append_basic_block('entry'))
        bldr.ret_void()
        set_cuda_kernel(kernel)

        fix_data_layout(m)
        ptx = llvm_to_ptx(str(m)).decode('utf8')
        self.assertTrue('mycudakernel' in ptx)
        if is64bit:
            self.assertTrue('.address_size 64' in ptx)
        else:
            self.assertTrue('.address_size 32' in ptx)

    def _test_nvvm_support(self, arch):
        nvvmir = self.get_ptx()
        compute_xx = 'compute_{0}{1}'.format(*arch)
        ptx = llvm_to_ptx(nvvmir, arch=compute_xx, ftz=1, prec_sqrt=0,
                          prec_div=0).decode('utf8')
        self.assertIn(".target sm_{0}{1}".format(*arch), ptx)
        self.assertIn('simple', ptx)
        self.assertIn('ave', ptx)

    def test_nvvm_support(self):
        """Test supported CC by NVVM
        """
        for arch in get_supported_ccs():
            self._test_nvvm_support(arch=arch)

    @unittest.skipIf(True, "No new CC unknown to NVVM yet")
    def test_nvvm_future_support(self):
        """Test unsupported CC to help track the feature support
        """
        # List known CC but unsupported by NVVM
        future_archs = [
            # (5, 2),  # for example
        ]
        for arch in future_archs:
            pat = r"-arch=compute_{0}{1}".format(*arch)
            with self.assertRaises(NvvmError) as raises:
                self._test_nvvm_support(arch=arch)
            self.assertIn(pat, raises.msg)


@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestArchOption(unittest.TestCase):
    def test_get_arch_option(self):
        # Test returning the nearest lowest arch.
        self.assertEqual(get_arch_option(5, 0), 'compute_50')
        self.assertEqual(get_arch_option(5, 1), 'compute_50')
        self.assertEqual(get_arch_option(3, 7), 'compute_35')
        # Test known arch.
        supported_cc = get_supported_ccs()
        for arch in supported_cc:
            self.assertEqual(get_arch_option(*arch), 'compute_%d%d' % arch)
        self.assertEqual(get_arch_option(1000, 0),
                         'compute_%d%d' % supported_cc[-1])


@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestLibDevice(unittest.TestCase):
    def _libdevice_load(self, arch, expect):
        libdevice = LibDevice(arch=arch)
        self.assertEqual(libdevice.arch, expect)

    def test_libdevice_arch_fix(self):
        self._libdevice_load('compute_20', 'compute_20')
        self._libdevice_load('compute_21', 'compute_20')
        self._libdevice_load('compute_30', 'compute_30')
        self._libdevice_load('compute_35', 'compute_35')
        self._libdevice_load('compute_52', 'compute_50')


gpu64 = '''
target triple="nvptx64-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @ave(i32 %a, i32 %b) {
entry:
%add = add nsw i32 %a, %b
%div = sdiv i32 %add, 2
ret i32 %div
}

define void @simple(i32* %data) {
entry:
%0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%mul = mul i32 %0, %1
%2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%add = add i32 %mul, %2
%call = call i32 @ave(i32 %add, i32 %add)
%idxprom = sext i32 %add to i64
%arrayidx = getelementptr inbounds i32, i32* %data, i64 %idxprom
store i32 %call, i32* %arrayidx, align 4
ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

!nvvm.annotations = !{!1}
!1 = metadata !{void (i32*)* @simple, metadata !"kernel", i32 1}
'''  # noqa: E501

gpu32 = '''
target triple="nvptx-"
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @ave(i32 %a, i32 %b) {
entry:
%add = add nsw i32 %a, %b
%div = sdiv i32 %add, 2
ret i32 %div
}

define void @simple(i32* %data) {
entry:
%0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%mul = mul i32 %0, %1
%2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%add = add i32 %mul, %2
%call = call i32 @ave(i32 %add, i32 %add)
%idxprom = sext i32 %add to i64
%arrayidx = getelementptr inbounds i32, i32* %data, i64 %idxprom
store i32 %call, i32* %arrayidx, align 4
ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

!nvvm.annotations = !{!1}
!1 = metadata !{void (i32*)* @simple, metadata !"kernel", i32 1}

'''  # noqa: E501

if __name__ == '__main__':
    unittest.main()
