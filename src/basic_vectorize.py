'''
Implements basic vectorize
'''

from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import numpy as np

class BasicUFunc(CDefinition):
    '''a generic ufunc that wraps the workload
    '''
    _argtys_ = [
        ('args',       C.pointer(C.char_p)),
        ('dimensions', C.pointer(C.intp)),
        ('steps',      C.pointer(C.intp)),
        ('data',       C.void_p),
    ]

    def body(self, args, dimensions, steps, data,):
        ufunc_ptr = self.depends(self.FuncDef)
        fnty = ufunc_ptr.type.pointee

        with self.for_range(dimensions[0]) as (loop, item):
            get_offset = lambda B, S, T: B[item * S].reference()\
                                                    .cast(C.pointer(T))

            indata = []
            for i, argty in enumerate(fnty.args):
                ptr = get_offset(args[i], steps[i], argty)
                indata.append(ptr.load())

            out_index = len(fnty.args)
            outptr = get_offset(args[out_index], steps[out_index],
                                fnty.return_type)

            res = ufunc_ptr(*indata)
            outptr.store(res)
        self.ret()

    @classmethod
    def specialize(cls, func_def):
        '''specialize to a workload
        '''
        cls._name_ = 'basicufunc_%s'% (func_def)
        cls.FuncDef = func_def


from parallel_vectorize import _llvm_ty_to_numpy
def basic_vectorize_from_func(lfunclist, engine=None):
    '''create ufunc from a llvm.core.Function

    lfunclist : a single or iterable of llvm.core.Function instance
    engine : [optional] a llvm.ee.ExecutionEngine instance

    If engine is given, return a function object which can be called
    from python.
    Otherwise, return the specialized ufunc(s) as a llvm.core.Function(s).
    '''
    import multiprocessing
    NUM_CPU = multiprocessing.cpu_count()

    try:
        iter(lfunclist)
    except TypeError:
        lfunclist = [lfunclist]

    buflist = []
    for lfunc in lfunclist:
        def_buf = BasicUFunc(CFuncRef(lfunc))
        buf = def_buf(lfunc.module)
        buflist.append(buf)

    if engine is None:
        # No engine given, just return the llvm definitions
        if len(buflist)==1:
            return buflist[0]
        else:
            return buflist

    # We have an engine, build ufunc
    from numbapro._internal import fromfunc

    try:
        ptr_t = long
    except:
        ptr_t = int
        assert False, "Have not check this yet" # Py3.0?

    ptrlist = []
    tyslist = []
    datlist = []
    for i, spuf in enumerate(buflist):
        fntype = lfunclist[i].type.pointee
        fptr = engine.get_pointer_to_function(spuf)
        argct = len(fntype.args)
        if i == 0: # for the first
            inct = argct
            outct = 1
        elif argct != inct:
            raise TypeError("All functions must have equal number of arguments")

        get_typenum = lambda T:np.dtype(_llvm_ty_to_numpy(T)).num
        assert fntype.return_type != C.void
        tys = list(map(get_typenum, list(fntype.args) + [fntype.return_type]))

        ptrlist.append(ptr_t(fptr))
        tyslist.append(tys)
        datlist.append(None)

    # Becareful that fromfunc does not provide full error checking yet.
    # If typenum is out-of-bound, we have nasty memory corruptions.
    # For instance, -1 for typenum will cause segfault.
    # If elements of type-list (2nd arg) is tuple instead,
    # there will also memory corruption. (Seems like code rewrite.)
    ufunc = fromfunc(ptrlist, tyslist, inct, outct, datlist)
    return ufunc
