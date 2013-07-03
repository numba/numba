import unittest
import numpy as np
from llvm_cbuilder import CExecutor, CDefinition
from llvm_cbuilder import shortnames as C
from llvm.ee import EngineBuilder
from llvm.core import Module
from numbapro.vectorizers.stream import stream_vectorize_from_func
from numbapro.vectorizers._common import _llvm_ty_to_dtype
from .support import addtest, main

class OneOne(CDefinition):

    def body(self, inval):
        self.ret( (inval * inval).cast(self.OUT_TYPE) )

    def specialize(cls, itype, otype):
        cls._name_ = '.'.join(map(str, ['oneone', itype, otype]))
        cls._retty_ = otype
        cls._argtys_ = [
            ('inval', itype),
        ]
        cls.OUT_TYPE = otype

@addtest
class TestStreamVectorize(unittest.TestCase):
    def test_streamvectorize_d_d(self):
        self.template(C.double, C.double)

    def test_streamvectorize_d_f(self):
        self.template(C.double, C.float)

    def test_streamvectorize_generic(self):
        module = Module.new(__name__)
        eb = EngineBuilder.new(module).mattrs('-avx').create()
        exe = CExecutor(eb)

        tyslist = [
            (C.double, C.double),
            (C.float,  C.float),
            (C.int64,  C.int64),
            (C.int32,  C.int32),
        ]

        tynumslist = []
        for tys in tyslist:
            tynumslist.append(list(map(_llvm_ty_to_dtype, tys)))


        oneone_defs = [OneOne(*tys)(module) for tys in tyslist]

        ufunc = stream_vectorize_from_func(oneone_defs, tynumslist, exe.engine,
                                           granularity=32)
        # print(module)
        module.verify()

        self.check(ufunc, np.float64)
        self.check(ufunc, np.float32)
        self.check(ufunc, np.int64)
        self.check(ufunc, np.int32)

    def check(self, ufunc, ty):
        x = np.linspace(0., 10., 500).astype(ty)

        ans = ufunc(x)
        gold = x * x

        for x, y in zip(ans, gold):
            if y != 0:
                err = abs(x - y)/y
                self.assertTrue(err < 1e-6)
            else:
                self.assertEqual(x, y)


    def template(self, itype, otype):
        module = Module.new(__name__)
        eb = EngineBuilder.new(module).mattrs('-avx').create()
        exe = CExecutor(eb)

        def_oneone = OneOne(itype, otype)
        oneone = def_oneone(module)

        tyslist = [list(map(_llvm_ty_to_dtype, [itype, otype]))]
        ufunc = stream_vectorize_from_func(oneone, tyslist, exe.engine,
                                           granularity=32)


        print(module)
        module.verify()
        print(module.to_native_assembly())

        self.check(ufunc, np.double)


if __name__ == '__main__':
    main()

