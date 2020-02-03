from numba import cuda
from numba.cuda import dispatcher
from numba.np.ufunc import deviceufunc

vectorizer_stager_source = '''
def __vectorized_{name}({args}, __out__):
    __tid__ = __cuda__.grid(1)
    if __tid__ < __out__.shape[0]:
        __out__[__tid__] = __core__({argitems})
'''


class CUDAVectorize(deviceufunc.DeviceVectorize):
    def _compile_core(self, sig):
        cudevfn = cuda.jit(sig, device=True, inline=True)(self.pyfunc)
        return cudevfn, cudevfn.cres.signature.return_type

    def _get_globals(self, corefn):
        glbl = self.pyfunc.__globals__.copy()
        glbl.update({'__cuda__': cuda,
                     '__core__': corefn})
        return glbl

    def _compile_kernel(self, fnobj, sig):
        return cuda.jit(fnobj)

    def build_ufunc(self):
        return dispatcher.CUDAUFuncDispatcher(self.kernelmap)

    @property
    def _kernel_template(self):
        return vectorizer_stager_source


# ------------------------------------------------------------------------------
# Generalized CUDA ufuncs

_gufunc_stager_source = '''
def __gufunc_{name}({args}):
    __tid__ = __cuda__.grid(1)
    if __tid__ < {checkedarg}:
        __core__({argitems})
'''


class CUDAGUFuncVectorize(deviceufunc.DeviceGUFuncVectorize):
    def build_ufunc(self):
        engine = deviceufunc.GUFuncEngine(self.inputsig, self.outputsig)
        return dispatcher.CUDAGenerializedUFunc(kernelmap=self.kernelmap,
                                                engine=engine)

    def _compile_kernel(self, fnobj, sig):
        return cuda.jit(sig)(fnobj)

    @property
    def _kernel_template(self):
        return _gufunc_stager_source

    def _get_globals(self, sig):
        corefn = cuda.jit(sig, device=True)(self.pyfunc)
        glbls = self.py_func.__globals__.copy()
        glbls.update({'__cuda__': cuda,
                      '__core__': corefn})
        return glbls
