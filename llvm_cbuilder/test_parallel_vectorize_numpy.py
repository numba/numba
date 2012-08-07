from test_parallel_vectorize import *

import numpy as np

def main():
    module = Module.new(__name__)
    PUfuncDef = ParallelUFuncPosix.specialize(num_thread=2)
    SPUF = SpecializedParallelUFunc.specialize(
                                        PUfuncDef, UFuncCore_D_D, Work_D_D)
    sppufunc = SPUF.define(module)

    module.verify()

    mpm = PassManager.new()
    pmbuilder = PassManagerBuilder.new()
    pmbuilder.opt_level = 3
    pmbuilder.populate(mpm)

    mpm.run(module)
#    print module

    # run

    exe = CExecutor(module)
    funcptr = exe.engine.get_pointer_to_function(sppufunc)
    print("Function pointer: %x" % funcptr)

    ptr_t = long # py2 only

    typenum = np.dtype(np.double).num
    ufunc = np.fromfunc([ptr_t(funcptr)], [[typenum, typenum]], 1, 1, [None])

    x = np.linspace(0., 10., 1000)
    x.dtype=np.double
#    print x
    ans = ufunc(x)
#    print ans

    if not ( ans == x/2.345 ).all():
        raise ValueError('Computation failed')
    else:
        print('Good')

if __name__ == '__main__':
    main()
