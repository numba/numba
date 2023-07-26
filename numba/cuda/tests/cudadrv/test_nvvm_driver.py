import warnings

from llvmlite import ir
from numba.cuda.cudadrv import nvvm
from ctypes import c_size_t, c_uint64, sizeof
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim

is64bit = sizeof(c_size_t) == sizeof(c_uint64)


@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestNvvmDriver(unittest.TestCase):
    def get_nvvmir(self):
        versions = NVVM().get_ir_version()
        metadata = metadata_nvvm70 % versions
        data_layout = NVVM().data_layout

        return nvvmir_generic.format(data_layout=data_layout, metadata=metadata)

    def test_nvvm_compile_simple(self):
        nvvmir = self.get_nvvmir()
        ptx = nvvm.llvm_to_ptx(nvvmir).decode('utf8')
        self.assertTrue('simple' in ptx)
        self.assertTrue('ave' in ptx)

    def test_nvvm_from_llvm(self):
        m = ir.Module("test_nvvm_from_llvm")
        m.triple = 'nvptx64-nvidia-cuda'
        nvvm.add_ir_version(m)
        fty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])
        kernel = ir.Function(m, fty, name='mycudakernel')
        bldr = ir.IRBuilder(kernel.append_basic_block('entry'))
        bldr.ret_void()
        nvvm.set_cuda_kernel(kernel)

        m.data_layout = NVVM().data_layout
        ptx = nvvm.llvm_to_ptx(str(m)).decode('utf8')
        self.assertTrue('mycudakernel' in ptx)
        if is64bit:
            self.assertTrue('.address_size 64' in ptx)
        else:
            self.assertTrue('.address_size 32' in ptx)

    def test_nvvm_ir_verify_fail(self):
        m = ir.Module("test_bad_ir")
        m.triple = "unknown-unknown-unknown"
        m.data_layout = NVVM().data_layout
        nvvm.add_ir_version(m)
        with self.assertRaisesRegex(NvvmError, 'Invalid target triple'):
            nvvm.llvm_to_ptx(str(m))

    def _test_nvvm_support(self, arch):
        compute_xx = 'compute_{0}{1}'.format(*arch)
        nvvmir = self.get_nvvmir()
        ptx = nvvm.llvm_to_ptx(nvvmir, arch=compute_xx, ftz=1, prec_sqrt=0,
                               prec_div=0).decode('utf8')
        self.assertIn(".target sm_{0}{1}".format(*arch), ptx)
        self.assertIn('simple', ptx)
        self.assertIn('ave', ptx)

    def test_nvvm_support(self):
        """Test supported CC by NVVM
        """
        for arch in nvvm.get_supported_ccs():
            self._test_nvvm_support(arch=arch)

    def test_nvvm_warning(self):
        m = ir.Module("test_nvvm_warning")
        m.triple = 'nvptx64-nvidia-cuda'
        m.data_layout = NVVM().data_layout
        nvvm.add_ir_version(m)

        fty = ir.FunctionType(ir.VoidType(), [])
        kernel = ir.Function(m, fty, name='inlinekernel')
        builder = ir.IRBuilder(kernel.append_basic_block('entry'))
        builder.ret_void()
        nvvm.set_cuda_kernel(kernel)

        # Add the noinline attribute to trigger NVVM to generate a warning
        kernel.attributes.add('noinline')

        with warnings.catch_warnings(record=True) as w:
            nvvm.llvm_to_ptx(str(m))

        self.assertEqual(len(w), 1)
        self.assertIn('overriding noinline attribute', str(w[0]))

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
        self.assertEqual(nvvm.get_arch_option(5, 3), 'compute_53')
        self.assertEqual(nvvm.get_arch_option(7, 5), 'compute_75')
        self.assertEqual(nvvm.get_arch_option(7, 7), 'compute_75')
        # Test known arch.
        supported_cc = nvvm.get_supported_ccs()
        for arch in supported_cc:
            self.assertEqual(nvvm.get_arch_option(*arch), 'compute_%d%d' % arch)
        self.assertEqual(nvvm.get_arch_option(1000, 0),
                         'compute_%d%d' % supported_cc[-1])


@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestLibDevice(unittest.TestCase):
    def test_libdevice_load(self):
        # Test that constructing LibDevice gives a bitcode file
        libdevice = LibDevice()
        self.assertEqual(libdevice.bc[:4], b'BC\xc0\xde')


nvvmir_generic = '''\
target triple="nvptx64-nvidia-cuda"
target datalayout = "{data_layout}"

define i32 @ave(i32 %a, i32 %b) {{
entry:
%add = add nsw i32 %a, %b
%div = sdiv i32 %add, 2
ret i32 %div
}}

define void @simple(i32* %data) {{
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
}}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

{metadata}
'''  # noqa: E501


metadata_nvvm70 = '''
!nvvmir.version = !{!1}
!1 = !{i32 %s, i32 %s, i32 %s, i32 %s}

!nvvm.annotations = !{!2}
!2 = !{void (i32*)* @simple, !"kernel", i32 1}
'''  # noqa: E501


metadata_nvvm34 = '''
!nvvm.annotations = !{!1}
!1 = metadata !{void (i32*)* @simple, metadata !"kernel", i32 1}
'''


if __name__ == '__main__':
    unittest.main()
