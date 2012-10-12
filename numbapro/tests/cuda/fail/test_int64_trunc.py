'''
As of LLVM3.1, PTX backend cannot handle truncation of 64-bit integer into
32-bit.  The LLVM-IR below will only work if the truncation:

  %17 = trunc i64 %12 to i32

is removed.

'''
import unittest
from nose.exc import SkipTest
from llvm.core import *
from llvm.ee import *

llvmir = '''; ModuleID = 'ptx_<function cu_array_double at 0x26c9758>'

declare i32 @llvm.ptx.read.tid.x() nounwind readnone

declare i32 @llvm.ptx.read.ctaid.x() nounwind readnone

declare i32 @llvm.ptx.read.ntid.x() nounwind readnone

define ptx_kernel void @ptxwrapper_cu_array_double0({ i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }*) {
entry:
  %1 = call i32 @llvm.ptx.read.tid.x()
  %2 = call i32 @llvm.ptx.read.ctaid.x()
  %3 = call i32 @llvm.ptx.read.ntid.x()
  %4 = mul i32 %2, %3
  %5 = add i32 %1, %4
  %6 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %0, i32 0, i32 2
  %7 = load i8** %6
  %8 = bitcast i8* %7 to i32*
  %9 = getelementptr i32* %8, i32 %5
  %10 = load i32* %9
  %11 = sext i32 %10 to i64
  %12 = mul i64 %11, 2
  %13 = getelementptr { i64, i32*, i8*, i32, i64*, i64*, i8*, i8*, i32, i8*, i8*, i8*, i64* }* %0, i32 0, i32 2
  %14 = load i8** %13
  %15 = bitcast i8* %14 to i32*
  %16 = getelementptr i32* %15, i32 %5
  %17 = trunc i64 %12 to i32
  store i32 %17, i32* %16
  ret void
}

'''

class TestCuda64bitTruncation(unittest.TestCase):
    def test_cuda_64_bit_trunc(self):
        raise SkipTest("Problem in LLVM 3.1")
        module = Module.from_assembly(llvmir)
        print module

        cc = 'sm_%d%d' % (2, 1)
        arch = 'ptx64'
        ptxtm = TargetMachine.lookup(arch, cpu=cc, opt=3)
        ptxasm = ptxtm.emit_assembly(module)

if __name__ == '__main__':
    unittest.main()

