from llvm.core import *
from llvm.passes import *
from llvm.ee import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import unittest, logging

def nestedloop1(d):
    z = 0
    for x in range(100):
        for y in range(100):
            z += x * d + int(y / d)
    return z

def gen_nestedloop1(mod):
    functype = Type.function(C.int, [C.int])
    func = mod.add_function(functype, 'nestedloop1')

    cb = CBuilder(func)

    d = cb.args[0]
    x = cb.var(C.int)
    y = cb.var(C.int)
    z = cb.var(C.int)

    one = cb.constant(C.int, 1)
    zero = cb.constant(C.int, 0)
    limit = cb.constant(C.int, 100)

    z.assign(zero)
    x.assign(zero)
    with cb.loop() as outer:
        with outer.condition() as setcond:
            setcond( x < limit )

        with outer.body():
            y.assign(zero)
            with cb.loop() as inner:
                with inner.condition() as setcond:
                    setcond( y < limit )

                with inner.body():
                    z += x * d + y / d
                    y += one
            x += one

    cb.ret(z)
    cb.close()
    return func


def nestedloop2(d):
    z = 0
    for x in range(1, 100):
        for y in range(1, 100):
            if x > y:
                z += int(x / y) * d
            else:
                z += int(y / x) * d
    return z

def gen_nestedloop2(mod):
    functype = Type.function(C.int, [C.int])
    func = mod.add_function(functype, 'nestedloop2')

    cb = CBuilder(func)

    d = cb.args[0]
    x = cb.var(C.int)
    y = cb.var(C.int)
    z = cb.var(C.int)

    one = cb.constant(C.int, 1)
    zero = cb.constant(C.int, 0)
    limit = cb.constant(C.int, 100)

    z.assign(zero)
    x.assign(one)
    with cb.loop() as outer:
        with outer.condition() as setcond:
            setcond( x < limit )

        with outer.body():
            y.assign(one)
            with cb.loop() as inner:
                with inner.condition() as setcond:
                    setcond( y < limit )

                with inner.body():
                    with cb.ifelse(x > y) as ifelse:
                        with ifelse.then():
                            z += x / y * d
                        with ifelse.otherwise():
                            z += y / x * d
                    y += one
            x += one

    cb.ret(z)
    cb.close()
    return func


class TestNestedLoop(unittest.TestCase):
    def test_nestedloop1(self):
        mod = Module.new(__name__)
        lfunc = gen_nestedloop1(mod)
        logging.debug(mod)
        mod.verify()

        exe = CExecutor(mod)
        func = exe.get_ctype_function(lfunc, 'int, int')
        for x in range(1, 100):
            self.assertEqual(func(x), int(nestedloop1(x)))

    def test_nestedloop2(self):
        mod = Module.new(__name__)
        lfunc = gen_nestedloop2(mod)
        logging.debug(mod)
        mod.verify()

        exe = CExecutor(mod)
        func = exe.get_ctype_function(lfunc, 'int, int')
        for x in range(1, 100):
            self.assertEqual(func(x), int(nestedloop2(x)))

if __name__ == '__main__':
    unittest.main()

