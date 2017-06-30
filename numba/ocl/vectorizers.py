from __future__ import print_function, absolute_import

from numba import ocl
from numba.npyufunc import deviceufunc
from . import dispatcher

vectorizer_stager_source = '''
def __vectorized_{name}({args}, __out__):
    __tid__ = __ocl__.get_global_id(0)
    if __tid__ < __out__.shape[0]:
        __out__[__tid__] = __core__({argitems})
'''


class OCLVectorize(deviceufunc.DeviceVectorize):
    def _compile_core(self, sig):
        cudevfn = ocl.jit(sig, device=True)(self.pyfunc)
        return cudevfn, cudevfn.cres.signature.return_type

    def _get_globals(self, corefn):
        glbl = self.pyfunc.__globals__.copy()
        glbl.update({'__ocl__': ocl,
                     '__core__': corefn})
        return glbl

    def _compile_kernel(self, fnobj, sig):
        return ocl.jit(fnobj)

    def build_ufunc(self):
        return dispatcher.OCLUFuncDispatcher(self.kernelmap)

    @property
    def _kernel_template(self):
        return vectorizer_stager_source


# ------------------------------------------------------------------------------
# Generalized OpenCL ufuncs

_gufunc_stager_source = '''
def __gufunc_{name}({args}):
    __tid__ = __ocl__.get_global_id(1)
    if __tid__ < {checkedarg}:
        __core__({argitems})
'''


class OCLGUFuncVectorize(deviceufunc.DeviceGUFuncVectorize):
    def build_ufunc(self):
        engine = deviceufunc.GUFuncEngine(self.inputsig, self.outputsig)
        return dispatcher.OCLGenerializedUFunc(kernelmap=self.kernelmap,
                                                engine=engine)

    def _compile_kernel(self, fnobj, sig):
        return ocl.jit(sig)(fnobj)

    @property
    def _kernel_template(self):
        return _gufunc_stager_source

    def _get_globals(self, sig):
        corefn = ocl.jit(sig, device=True)(self.pyfunc)
        glbls = self.py_func.__globals__.copy()
        glbls.update({'__ocl__': ocl,
                      '__core__': corefn})
        return glbls
