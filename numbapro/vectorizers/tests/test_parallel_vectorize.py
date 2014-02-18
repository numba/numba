# import unittest
# from llvm.core import Module
# from llvm.passes import PassManager, PassManagerBuilder
# from llvm.ee import EngineBuilder
# from llvm_cbuilder import CExecutor, CDefinition
# from llvm_cbuilder import shortnames as C
# from numbapro.vectorizers.parallel import (
#     SpecializedParallelUFunc,
#     ParallelUFuncPlatform,
#     UFuncCoreGeneric,
#     )
# from .support import addtest, main
#
# class Work_D_D(CDefinition):
#     _name_ = 'work_d_d'
#     _retty_ = C.double
#     _argtys_ = [
#         ('inval', C.double),
#     ]
#     def body(self, inval):
#         self.ret(inval / self.constant(inval.type, 2.345))

# @addtest
# class Tester(CDefinition):
#     '''
#     Generate test.
#     '''
#     _name_ = 'tester'
#
#     def body(self):
#         # depends
#         module = self.function.module
#
#         ThreadCount = 2
#         ArgCount = 2
#         WorkCount = 1000
#
#         lfunc = Work_D_D()(module)
#         spufdef = SpecializedParallelUFunc(ParallelUFuncPlatform(num_thread=2),
#                                            UFuncCoreGeneric(lfunc))
#
#         sppufunc = self.depends(spufdef)
#
#         # real work
#         NULL = self.constant_null(C.void_p)
#
#         args = self.array(C.char_p, 2, name='args')
#
#         args_double = []
#         for t in range(ThreadCount):
#             args_for_thread = self.array(C.double, WorkCount)
#             args[t].assign(args_for_thread.cast(C.char_p))
#             args_double.append(args_for_thread)
#
#         dims = self.array(C.intp, 1, name='dims')
#         dims[0].assign(self.constant(C.intp, WorkCount))
#
#         steps = self.array(C.intp, ArgCount, name='steps')
#
#         for c in range(ArgCount):
#             steps[c].assign(self.constant(C.intp, 8))
#
#         # populate data
#         inbase = args_double[0]
#
#         i = self.var(C.intp, 0)
#         with self.loop() as loop:
#             with loop.condition() as setcond:
#                 setcond( i < self.constant(C.intp, WorkCount) )
#             with loop.body():
#                 inbase[i].assign(i.cast(C.double))
#                 i += self.constant(C.intp, 1)
#
#         # call parallel ufunc
#         sppufunc(args, dims, steps, NULL)
#
#         # check error
#         outbase = args_double[-1]
#         with self.for_range(self.constant(C.intp, WorkCount)) as (loop, i):
#             test = outbase[i] != (inbase[i] / self.constant(C.double, 2.345))
#             with self.ifelse( test ) as ifelse:
#                 with ifelse.then():
#                     self.debug("Invalid data at i =", i, outbase[i], inbase[i])
#
#         self.ret()
#
# class TestInner_RaceCondition(unittest.TestCase):
#     def test_racecondition(self):
#         module = Module.new(__name__)
#
#         mpm = PassManager.new()
#         pmbuilder = PassManagerBuilder.new()
#         pmbuilder.opt_level = 3
#         pmbuilder.populate(mpm)
#
#         fntester = Tester().define(module)
#
#         #    print(module)
#         module.verify()
#
#         mpm.run(module)
#
#         #   print('optimized'.center(80,'-'))
#         #   print(module)
#
#         # run
#         #   print('run')
#
#         eb = EngineBuilder.new(module).mattrs('-avx').create()
#         exe = CExecutor(eb)
#         func = exe.get_ctype_function(fntester, 'void')
#
#         func()
#         # Will not reach here if race condition occurred
#         #   print('Good')

if __name__ == '__main__':
    main()

