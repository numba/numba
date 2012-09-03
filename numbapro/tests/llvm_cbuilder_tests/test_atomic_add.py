'''
Base on the test_pthread.py and extend to use atomic instructions
'''

from llvm.core import *
from llvm.passes import *
from llvm.ee import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import unittest, logging
import sys

# logging.basicConfig(level=logging.DEBUG)

NUM_OF_THREAD = 4
REPEAT = 10000

def gen_test_worker(mod):
    cb = CBuilder.new_function(mod, 'worker', C.void, [C.pointer(C.int)])
    pval = cb.args[0]
    one = cb.constant(pval.type.pointee, 1)

    ct = cb.var(C.int, 0)
    limit = cb.constant(C.int, REPEAT)
    with cb.loop() as loop:
        with loop.condition() as setcond:
            setcond( ct < limit )

        with loop.body():
            cb.atomic_add(pval, one, 'acq_rel')
            ct += one

    cb.ret()
    cb.close()
    return cb.function

def gen_test_pthread(mod):
    cb = CBuilder.new_function(mod, 'manager', C.int, [C.int])
    arg = cb.args[0]

    worker_func = cb.get_function_named('worker')
    pthread_create = cb.get_function_named('pthread_create')
    pthread_join = cb.get_function_named('pthread_join')


    NULL = cb.constant_null(C.void_p)
    cast_to_null = lambda x: x.cast(C.void_p)

    threads = cb.array(C.void_p, NUM_OF_THREAD)

    for tid in range(NUM_OF_THREAD):
        pthread_create_args = [threads[tid].reference(),
                               NULL,
                               worker_func,
                               arg.reference()]
        pthread_create(*map(cast_to_null, pthread_create_args))

    worker_func(arg.reference())

    for tid in range(NUM_OF_THREAD):
        pthread_join_args = threads[tid], NULL
        pthread_join(*map(cast_to_null, pthread_join_args))


    cb.ret(arg)
    cb.close()
    return cb.function

class TestAtomicAdd(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32', "test uses pthreads, not supported on Windows")
    def test_atomic_add(self):
        mod = Module.new(__name__)
        # add pthread functions

        mod.add_function(Type.function(C.int,
                                       [C.void_p, C.void_p, C.void_p, C.void_p]),
                         'pthread_create')

        mod.add_function(Type.function(C.int,
                                       [C.void_p, C.void_p]),
                         'pthread_join')

        lf_test_worker = gen_test_worker(mod)
        lf_test_pthread = gen_test_pthread(mod)
        logging.debug(mod)
        mod.verify()

        # optimize
        fpm = FunctionPassManager.new(mod)
        mpm = PassManager.new()
        pmb = PassManagerBuilder.new()
        pmb.vectorize = True
        pmb.opt_level = 3
        pmb.populate(fpm)
        pmb.populate(mpm)

        fpm.run(lf_test_worker)
        fpm.run(lf_test_pthread)
        mpm.run(mod)
        logging.debug(mod)
        mod.verify()

        # run
        exe = CExecutor(mod)
        exe.engine.get_pointer_to_function(mod.get_function_named('worker'))
        func = exe.get_ctype_function(lf_test_pthread, 'int, int')

        inarg = 1234
        gold = inarg + (NUM_OF_THREAD + 1) * REPEAT

        for _ in range(1000): # run many many times to catch race condition
            self.assertEqual(func(inarg), gold, "Unexpected race condition")


if __name__ == '__main__':
    unittest.main()

