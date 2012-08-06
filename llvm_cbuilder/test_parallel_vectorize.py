from parallel_vectorize import *

'''
void parallel_ufunc(void * func, void * worker,
                    char **args, npy_intp *dimensions,
                    npy_intp *steps, void *data)
'''
class Tester(CDefinition):
    _name_ = 'tester'

    def body(self):
        # depends
        module = self.function.module

        ThreadCount = 2
        ArgCount = 2
        WorkCount = 1000000

        parallel_ufunc = CFunc(self, ParallelUFuncPosix.define(module,
                               ThreadCount=ThreadCount))
        worker = CFunc(self, UFuncCore.define(module))

        # real work
        NULL = self.constant_null(C.void_p)

        args = self.array(C.char_p, 2, name='args')

        for t in range(ThreadCount):
            args_for_thread = self.array(C.double, 4)
            args[t].assign(args_for_thread.cast(C.char_p))

        dims = self.array(C.intp, 1, name='dims')
        dims[0].assign(self.constant(C.intp, WorkCount))

        steps = self.array(C.intp, ArgCount, name='steps')

        for c in range(ArgCount):
            steps[c].assign(self.constant(C.intp, 8))

        parallel_ufunc(NULL, worker.cast(C.void_p),  args, dims, steps, NULL)

        self.ret()

def main():
    NUM_THREAD = 2
    module = Module.new(__name__)

    mpm = PassManager.new()
    pmbuilder = PassManagerBuilder.new()
    pmbuilder.opt_level = 3
    pmbuilder.populate(mpm)

    fntester = Tester.define(module)

#    print(module)
    module.verify()

    mpm.run(module)

    print('optimized'.center(80,'-'))
    print(module)

    # run
    exe = CExecutor(module)
    exe.engine.get_pointer_to_function(fntester)
    func = exe.get_ctype_function(fntester, 'void')

    func()


if __name__ == '__main__':
    main()

