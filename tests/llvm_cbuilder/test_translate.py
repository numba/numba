from llvm.core import Module
from llvm_cbuilder import *
from llvm_cbuilder.translator import translate
import llvm_cbuilder.shortnames as C
import unittest, logging

class Foo(CDefinition):
    _name_ = 'foo'
    _retty_ = C.int
    _argtys_ = [('x', C.int),
                ('y', C.int),]

    def body(self, x, y):
        @translate
        def _():
            if x > y:
                return x - y
            else:
                return y - x
        self.unreachable()

class TestTranslate(unittest.TestCase):
    def test_loopbreak(self):
        mod = Module.new(__name__)
        lfoo = Foo()(mod)

        print(mod)
        mod.verify()

        exe = CExecutor(mod)
        foo = exe.get_ctype_function(lfoo, 'int, int')
        print(foo(10, 20))

if __name__ == '__main__':
    unittest.main()

