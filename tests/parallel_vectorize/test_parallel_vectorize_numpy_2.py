'''
Test parallel-vectorize with numpy.fromfunc.
Uses the work load from test_parallel_vectorize.
This time we pass a function pointer.
'''

from test_parallel_vectorize import *
from numbapro._internal import fromfunc
import numpy as np

def main():
    module = Module.new(__name__)
    exe = CExecutor(module)

    workdef = Work_D_D()
    workfunc = workdef(module)

    # get pointer to workfunc
    workfunc_ptr = exe.engine.get_pointer_to_function(workfunc)

    workdecl = CFuncRef(workfunc.name, workfunc.type.pointee, workfunc_ptr)

    spufdef = SpecializedParallelUFunc(ParallelUFuncPosix(num_thread=2),
                                       UFuncCore_D_D(),
                                       workdecl)


    sppufunc = spufdef(module)
    sppufunc.verify()
    print(sppufunc)
    module.verify()

    mpm = PassManager.new()
    pmbuilder = PassManagerBuilder.new()
    pmbuilder.opt_level = 3
    pmbuilder.populate(mpm)

    mpm.run(module)
    print(module)


    # run

    funcptr = exe.engine.get_pointer_to_function(sppufunc)
    print("Function pointer: %x" % funcptr)

    ptr_t = long # py2 only

    # Becareful that fromfunc does not provide full error checking yet.
    # If typenum is out-of-bound, we have nasty memory corruptions.
    # For instance, -1 for typenum will cause segfault.
    # If elements of type-list (2nd arg) is tuple instead,
    # there will also memory corruption. (Seems like code rewrite.)
    typenum = np.dtype(np.double).num
    ufunc = fromfunc([ptr_t(funcptr)], [[typenum, typenum]], 1, 1, [None])

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
