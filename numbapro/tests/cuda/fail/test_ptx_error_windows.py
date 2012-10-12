import unittest
from nose.exc import SkipTest
from llvm.core import *
from llvm.ee import *

class TestPTXErrorOnWindows(unittest.TestCase):
    def test_ptx_error_on_windows(self):
        raise SkipTest("This checks if the PTX error on windows is fixed."
                       "This runs fine on Linux.")
        with open("./test_ptx_error_windows.ll", "rb") as fin:
            m = Module.from_assembly(fin)
        tm = TargetMachine.lookup('ptx64', 'sm_21')
        ptx = tm.emit_assembly(m)
        print ptx
        # There will be non-ascii characters on Windows
        # around:
        #       @%p2	bra	$L__BB0_1;
	    #           ld.global.u64	%rd5, [%rd41];
        self.assertTrue(all(ord(c) < 128 and ord(c) >=0 for c in ptx))

if __name__ == '__main__':
    unittest.main()
