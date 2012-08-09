from parallel_vectorize import *
from llvm_cbuilder import shortnames as C
from llvm.core import *
import numpy as np
import unittest
from random import random

class OneOne(CDefinition):

    def body(self, inval):
        self.ret( (inval * inval).cast(self.OUT_TYPE) )

    @classmethod
    def specialize(cls, itype, otype):
        cls._name_ = '.'.join(map(str, ['oneone', itype, otype]))
        cls._retty_ = otype
        cls._argtys_ = [
            ('inval', itype),
        ]
        cls.OUT_TYPE = otype


class TestParallelVectorize(unittest.TestCase):
    def test_parallelvectorize_d_d(self):
        self.template(C.double, C.double)

    def test_parallelvectorize_d_f(self):
        self.template(C.double, C.float)

    def template(self, itype, otype):
        module = Module.new(__name__)
        exe = CExecutor(module)

        def_oneone = OneOne(itype, otype)
        oneone = def_oneone(module)
        ufunc = parallel_vectorize_from_func(oneone, exe.engine)
        # print(module)
        module.verify()

        x = np.linspace(.0, 10., 1000)
        x.dtype = np.double

        ans = ufunc(x)
        gold = x * x

        for x, y in zip(ans, gold):
            if y != 0:
                err = abs(x - y)/y
                self.assertLess(err, 1e-6)
            else:
                self.assertEqual(x, y)


if __name__ == '__main__':
    unittest.main()

