from numbapro.vectorizers.parallel import *
from numbapro.vectorizers._common import _llvm_ty_to_dtype
from llvm_cbuilder import shortnames as C
from llvm.core import *
from llvm.ee import EngineBuilder
import numpy as np
import unittest
from random import random
from .support import addtest, main


class TwoOne(CDefinition):

    def body(self, a, b):
        self.ret( (a * b).cast(self.OUT_TYPE) )

    def specialize(cls, itype1, itype2, otype):
        cls._name_ = '.'.join(map(str, ['oneone', itype1, itype2, otype]))
        cls._retty_ = otype
        cls._argtys_ = [
            ('a', itype1),
            ('b', itype2),
        ]
        cls.OUT_TYPE = otype

@addtest
class TestParallelVectorize(unittest.TestCase):
    def test_parallelvectorize_dd_d(self):
        self.template(C.double, C.double, C.double)

    def test_parallelvectorize_dd_f(self):
        self.template(C.double, C.double, C.float)

    def test_parallelvectorize_generic(self):
        module = Module.new(__name__)

        eb = EngineBuilder.new(module).mattrs('-avx').create()
        exe = CExecutor(eb)

        tyslist = [
            (C.double, C.double, C.double),
            (C.float,  C.float,  C.float),
            (C.int64,  C.int64,  C.int64),
            (C.int32,  C.int32,  C.int32),
        ]


        tynumslist = []
        for tys in tyslist:
            tynumslist.append(list(map(_llvm_ty_to_dtype, tys)))


        twoone_defs = [TwoOne(*tys)(module) for tys in tyslist]

        ufunc = parallel_vectorize_from_func(twoone_defs, tynumslist, exe.engine)
        # print(module)
        module.verify()

        self.check(ufunc, np.double)
        self.check(ufunc, np.float32)
        self.check(ufunc, np.int64)
        self.check(ufunc, np.int32)

    def check(self, ufunc, ty):
        A = np.linspace(.0, 10., 1000).astype(ty)
        B = np.linspace(-10., 0., 1000).astype(ty)

        ans = ufunc(A, B)
        gold = A * B

        for x, y in zip(ans, gold):
            if y != 0:
                err = abs(x - y)/y
                self.assertTrue(err < 1e-6)
            else:
                self.assertEqual(x, y)

    def template(self, itype1, itype2, otype):
        module = Module.new(__name__)

        eb = EngineBuilder.new(module).mattrs('-avx').create()
        exe = CExecutor(eb)

        def_twoone = TwoOne(itype1, itype2, otype)
        twoone = def_twoone(module)

        tyslist = [list(map(_llvm_ty_to_dtype, [itype1, itype2, otype]))]
        ufunc = parallel_vectorize_from_func(twoone, tyslist, exe.engine)
        # print(module)
        module.verify()

        self.check(ufunc, np.double)

if __name__ == '__main__':
    main()

