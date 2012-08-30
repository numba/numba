from llvm.core import *
from llvm.passes import *
from llvm.ee import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import unittest, logging

def is_prime(x):
    if x <= 2:
        return True
    if (x % 2) == 0:
        return False
    for y in range(2, int(1 + x**0.5)):
        if (x % y) == 0:
            return False
    return True

def gen_is_prime(mod):
    functype = Type.function(C.int, [C.int])
    func = mod.add_function(functype, 'isprime')

    cb = CBuilder(func)

    arg = cb.args[0]

    two = cb.constant(C.int, 2)
    true = one = cb.constant(C.int, 1)
    false = zero = cb.constant(C.int, 0)

    with cb.ifelse( arg <= two ) as ifelse:
        with ifelse.then():
            cb.ret(true)

    with cb.ifelse( (arg % two) == zero ) as ifelse:
        with ifelse.then():
            cb.ret(false)

    idx = cb.var(C.int, 3, name='idx')
    with cb.loop() as loop:
        with loop.condition() as setcond:
            setcond( idx < arg )

        with loop.body():
            with cb.ifelse( (arg % idx) == zero ) as ifelse:
                with ifelse.then():
                    cb.ret(false)
            # increment
            idx += two

    cb.ret(true)
    cb.close()
    return func


def gen_is_prime_fast(mod):
    functype = Type.function(C.int, [C.int])
    func = mod.add_function(functype, 'isprime_fast')

    cb = CBuilder(func)

    arg = cb.args[0]

    two = cb.constant(C.int, 2)
    true = one = cb.constant(C.int, 1)
    false = zero = cb.constant(C.int, 0)

    with cb.ifelse( arg <= two ) as ifelse:
        with ifelse.then():
            cb.ret(true)

    with cb.ifelse( (arg % two) == zero ) as ifelse:
        with ifelse.then():
            cb.ret(false)

    idx = cb.var(C.int, 3, name='idx')

    sqrt = cb.get_intrinsic(INTR_SQRT, [C.float])

    looplimit = one + sqrt(arg.cast(C.float)).cast(C.int)


    with cb.loop() as loop:
        with loop.condition() as setcond:
            setcond( idx < looplimit )

        with loop.body():
            with cb.ifelse( (arg % idx) == zero ) as ifelse:
                with ifelse.then():
                    cb.ret(false)
            # increment
            idx += two


    cb.ret(true)
    cb.close()
    return func

class TestIsPrime(unittest.TestCase):
    def test_isprime(self):
        mod = Module.new(__name__)
        lf_isprime = gen_is_prime(mod)
        logging.debug(mod)
        mod.verify()

        exe = CExecutor(mod)
        func = exe.get_ctype_function(lf_isprime, 'bool, int')
        for x in range(2, 1000):
            msg = "Failed at x = %d" % x
            self.assertEqual(func(x), is_prime(x), msg)

    def test_isprime_fast(self):
        mod = Module.new(__name__)
        lf_isprime = gen_is_prime_fast(mod)
        logging.debug(mod)
        mod.verify()

        exe = CExecutor(mod)
        func = exe.get_ctype_function(lf_isprime, 'bool, int')
        for x in range(2, 1000):
            msg = "Failed at x = %d" % x
            self.assertEqual(func(x), is_prime(x), msg)

if __name__ == '__main__':
    unittest.main()

