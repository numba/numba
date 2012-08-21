import numpy as np
from llvm_cbuilder import shortnames as _C

_llvm_ty_str_to_numpy = {
            'i8'     : np.int8,
            'i16'    : np.int16,
            'i32'    : np.int32,
            'i64'    : np.int64,
            'float'  : np.float32,
            'double' : np.float64,
        }

def _llvm_ty_to_numpy(ty):
    return _llvm_ty_str_to_numpy[str(ty)]

class CommonVectorizeFromFrunc(object):
    def build(self, lfunc):
        raise NotImplementedError

    def __call__(self, lfunclist, engine=None):
        '''create ufunc from a llvm.core.Function

        lfunclist : a single or iterable of llvm.core.Function instance
        engine : [optional] a llvm.ee.ExecutionEngine instance

        If engine is given, return a function object which can be called
        from python.
        Otherwise, return the specialized ufunc(s) as a llvm.core.Function(s).
        '''
        try:
            iter(lfunclist)
        except TypeError:
            lfunclist = [lfunclist]

        self.lfunclist = lfunclist

        # build all functions
        spuflist = [self.build(lfunc) for lfunc in lfunclist]

        if engine is None:
            # No engine given, just return the llvm definitions
            if len(spuflist)==1:
                return spuflist[0]
            else:
                return spuflist

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
        for i, spuf in enumerate(spuflist):
            fntype = lfunclist[i].type.pointee
            fptr = engine.get_pointer_to_function(spuf)
            argct = len(fntype.args)
            if i == 0: # for the first
                inct = argct
                outct = 1
            elif argct != inct:
                raise TypeError("All functions must have equal number of arguments")

            get_typenum = lambda T:np.dtype(_llvm_ty_to_numpy(T)).num
            assert fntype.return_type != _C.void
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


from numbapro.translate import Translate

class GenericVectorize(object):
    def __init__(self, func):
        self.pyfunc = func
        self.translates = []

    def add(self, *args, **kwargs):
        t = Translate(self.pyfunc, *args, **kwargs)
        t.translate()
        self.translates.append(t)

    def _get_lfunc_list(self):
        return [t.lfunc for t in self.translates]

    def build_ufunc(self):
        raise NotImplementedError

def ufunc_core_impl(fnty, func, args, steps, item):
    get_offset = lambda B, S, T: B[item * S].reference().cast(_C.pointer(T))

    indata = []
    for i, argty in enumerate(fnty.args):
        ptr = get_offset(args[i], steps[i], argty)
        indata.append(ptr.load())

    out_index = len(fnty.args)
    outptr = get_offset(args[out_index], steps[out_index],
                        fnty.return_type)

    res = func(*indata)
    outptr.store(res)


