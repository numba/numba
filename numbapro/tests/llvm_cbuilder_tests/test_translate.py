
from llvm.core import Module
from llvm_cbuilder import *
from llvm_cbuilder.translator import translate
import llvm_cbuilder.shortnames as C
import unittest, logging

#logging.basicConfig(level=logging.DEBUG)

class FooIf(CDefinition):
    _name_ = 'foo_if'
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


class FooWhile(CDefinition):
    _name_ = 'foo_while'
    _retty_ = C.int
    _argtys_ = [('x', C.int)]

    def body(self, x):
        y = self.var_copy(x)

        @translate
        def _():
            while x > 0:
                x -= 1
                y += x
            return y

class FooForRange(CDefinition):
    _name_ = 'foo_for_range'
    _retty_ = C.int
    _argtys_ = [('x', C.int)]

    def body(self, x):
        y = self.var(x.type, 0)

        @translate
        def _():
            for i in range(x + 1):
                y += i
            return y


class TestTranslate(unittest.TestCase):
    def test_if(self):
        mod = Module.new(__name__)
        lfoo = FooIf()(mod)

        print(mod)
        mod.verify()

        exe = CExecutor(mod)
        foo = exe.get_ctype_function(lfoo, 'int, int')
        self.assertEqual(foo(10, 20), 20 - 10)
        self.assertEqual(foo(23, 17), 23 - 17)

    def test_whileloop(self):
        mod = Module.new(__name__)
        lfoo = FooWhile()(mod)

        print(mod)
        mod.verify()

        exe = CExecutor(mod)
        foo = exe.get_ctype_function(lfoo, 'int')
        self.assertEqual(foo(10), sum(range(10+1)))
        self.assertEqual(foo(1324), sum(range(1324+1)))

    def test_forloop(self):
        mod = Module.new(__name__)
        lfoo = FooForRange()(mod)

        print(mod)
        mod.verify()

        exe = CExecutor(mod)
        foo = exe.get_ctype_function(lfoo, 'int')
        self.assertEqual(foo(10), sum(range(10+1)))
        self.assertEqual(foo(1324), sum(range(1324+1)))

if __name__ == '__main__':
    unittest.main()

