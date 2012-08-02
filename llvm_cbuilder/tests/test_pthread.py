from llvm.core import *
from llvm.passes import *
from llvm.ee import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import unittest, logging

logging.basicConfig(level=logging.DEBUG)

NUM_OF_THREAD = 4

def gen_test_worker(mod):
    cb = CBuilder.new_function(mod, 'worker', C.void, [C.pointer(C.int)])
    pval = cb.args[0]
    val = pval.load()
    one = cb.constant(val.type, 1)
    pval.store(val + one)
    cb.ret()
    cb.close()

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

    for tid in range(NUM_OF_THREAD):
        pthread_join_args = threads[tid], NULL
        pthread_join(*map(cast_to_null, pthread_join_args))

    worker_func(arg.reference())

    cb.ret(arg)
    cb.close()
    return cb.function

class TestPThread(unittest.TestCase):
    def test_pthread(self):
        mod = Module.new(__name__)
        # add pthread functions

        mod.add_function(Type.function(C.int,
                                       [C.void_p, C.void_p, C.void_p, C.void_p]),
                         'pthread_create')

        mod.add_function(Type.function(C.int,
                                       [C.void_p, C.void_p]),
                         'pthread_join')

        gen_test_worker(mod)
        lf_test_pthread = gen_test_pthread(mod)
        logging.debug(mod)
        mod.verify()

        exe = CExecutor(mod)
        exe.engine.get_pointer_to_function(mod.get_function_named('worker'))
        func = exe.get_ctype_function(lf_test_pthread, 'int, int')

        inarg = 1234
        self.assertEqual(func(inarg), inarg+NUM_OF_THREAD+1)

if __name__ == '__main__':
    unittest.main()

