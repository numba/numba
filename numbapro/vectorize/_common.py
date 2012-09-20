import numpy as np
from llvm_cbuilder import shortnames as _C
from numbapro import _internal
from numbapro.translate import Translate
from llvm.passes import PassManager, PassManagerBuilder

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

def _llvm_ty_to_dtype_num(ty):
    return np.dtype(_llvm_ty_to_numpy(ty)).num

_numbatypes_str_to_numpy = {
            'int8'     : np.int8,
            'int16'    : np.int16,
            'int32'    : np.int32,
            'int64'    : np.int64,
            'uint8'    : np.uint8,
            'uint16'   : np.uint16,
            'uint32'   : np.uint32,
            'uint64'   : np.uint64,
#            'f'        : np.float32,
#            'd'        : np.float64,
            'float'    : np.float32,
            'double'   : np.float64,
        }

def _numbatypes_to_numpy(ty):
    ret = _numbatypes_str_to_numpy[str(ty)]
    return ret

class CommonVectorizeFromFrunc(object):
    def build(self, lfunc):
        raise NotImplementedError

    def __call__(self, lfunclist, tyslist, engine,
                 minivect_dispatcher=None, cuda_dispatcher=None,
                 **kws):
        '''create ufunc from a llvm.core.Function

        lfunclist : a single or iterable of llvm.core.Function instance
        engine : a llvm.ee.ExecutionEngine instance

        return a function object which can be called from python.
        '''
        try:
            iter(lfunclist)
        except TypeError:
            lfunclist = [lfunclist]


        ptrlist = self._prepare_pointers(lfunclist, engine, **kws)

        fntype = lfunclist[0].type.pointee
        inct = len(fntype.args)
        outct = 1

        datlist = [None] * len(lfunclist)

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)
        ufunc = _internal.fromfunc(ptrlist, tyslist, inct, outct,
                                   datlist, minivect_dispatcher,
                                   cuda_dispatcher)
        return ufunc

    def _prepare_pointers(self, lfunclist, engine, **kws):
        # build all functions
        # spuflist = [self.build(lfunc, **kws) for lfunc in lfunclist]
        spuflist = []
        for i, lfunc in enumerate(lfunclist):
            spuflist.append(self.build(lfunc, **kws))

        # We have an engine, build ufunc

        try:
            ptr_t = long
        except:
            ptr_t = int
            assert False, "Have not check this yet" # Py3.0?

        ptrlist = []
        tyslist = []
        datlist = []
        for i, spuf in enumerate(spuflist):
            fptr = engine.get_pointer_to_function(spuf)
            ptrlist.append(ptr_t(fptr))

        return ptrlist

class GenericVectorize(object):
    def __init__(self, func):
        self.pyfunc = func
        self.translates = []
        self.args_ret_types = []

    def add(self, *args, **kwargs):
        t = Translate(self.pyfunc, *args, **kwargs)
        t.translate()
        self.translates.append(t)

        argtys = kwargs['arg_types']
        retty = kwargs['ret_type']
        self.args_ret_types.append(argtys + [retty])

    def _get_tys_list(self):
        tyslist = []
        for args_ret in self.args_ret_types:
            tys = []
            for ty in args_ret:
                tys.append(np.dtype(_numbatypes_to_numpy(ty)).num)
            tyslist.append(tys)
        return tyslist

    def _get_lfunc_list(self):
        return [t.lfunc for t in self.translates]

    def build_ufunc(self):
        raise NotImplementedError

def post_vectorize_optimize(func):
    '''Perform aggressive optimization after each vectorizer.

    TODO: Currently uses Module level PassManager each is rather wasteful
          and may have side-effect on other already optimized functions.
          We should find out a list of optimization to add use in
          FunctionPassManager.
    '''
    pmb = PassManagerBuilder.new()
    pmb.opt_level = 3
    pmb.vectorize = True

    pm = PassManager.new()
    pmb.populate(pm)

    pm.run(func.module)

