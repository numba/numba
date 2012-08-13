from llvm.core import *
from llvm.passes import *
from llvm.ee import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import unittest, logging

def loopbreak(d):
    z = 0
    for x in range(100):
        for y in range(100):
            z += x + y
            if z > 50:
                break
        z -= d
    return z

def gen_loopbreak(mod):
    functype = Type.function(C.int, [C.int])
    func = mod.add_function(functype, 'loopbreak')

    cb = CBuilder(func)

    d = cb.args[0]
    x = cb.var(C.int)
    y = cb.var(C.int)
    z = cb.var(C.int)

    one = cb.constant(C.int, 1)
    zero = cb.constant(C.int, 0)
    limit = cb.constant(C.int, 100)
    fifty = cb.constant(C.int, 50)

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
                    z += x + y
                    with cb.ifelse( z > fifty ) as ifelse:
                        with ifelse.then():
                            inner.break_loop()
                    y += one
            z -= d
            x += one

    cb.ret(z)
    cb.close()
    return func

def loopcontinue(d):
    z = 0
    for x in range(100):
        for y in range(100):
            z += x + y
            if z > 50:
                continue
            z += d
    return z

def gen_loopcontinue(mod):
    functype = Type.function(C.int, [C.int])
    func = mod.add_function(functype, 'loopcontinue')

    cb = CBuilder(func)

    d = cb.args[0]
    x = cb.var(C.int)
    y = cb.var(C.int)
    z = cb.var(C.int)

    one = cb.constant(C.int, 1)
    zero = cb.constant(C.int, 0)
    limit = cb.constant(C.int, 100)
    fifty = cb.constant(C.int, 50)

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
                    z += x + y
                    y += one
                    with cb.ifelse( z > fifty ) as ifelse:
                        with ifelse.then():
                            inner.continue_loop()
                    z += d
            x += one

    cb.ret(z)
    cb.close()
    return func

class TestLoopControl(unittest.TestCase):
    def test_loopbreak(self):
        mod = Module.new(__name__)
        lfunc = gen_loopbreak(mod)
        logging.debug(mod)
        mod.verify()

        exe = CExecutor(mod)
        func = exe.get_ctype_function(lfunc, 'int, int')
        for x in range(100):
            self.assertEqual(func(x), loopbreak(x))

    def test_loopcontinue(self):
        mod = Module.new(__name__)
        lfunc = gen_loopcontinue(mod)
        logging.debug(mod)
        mod.verify()

        exe = CExecutor(mod)
        func = exe.get_ctype_function(lfunc, 'int, int')
        for x in range(100):
            self.assertEqual(func(x), loopcontinue(x))

if __name__ == '__main__':
    unittest.main()

