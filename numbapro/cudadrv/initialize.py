from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from numba.targets.registry import target_registry
from numba.npyufunc import Vectorize, GUVectorize
#
# Public
#
last_error = None

_is_initialize = False

def initialize():
    "Safe to run multiple times"
    global _is_initialize
    if _is_initialize: return
    from .error import CudaSupportError, NvvmSupportError
    global last_error
    try:
        _init_driver()
        _init_nvvm()
        _init_numba_jit_registry()
        _is_initialize = True
        return True
    except CudaSupportError, e:
        last_error = e
        _init_poison_jit_registry()
        return False
    except NvvmSupportError, e:
        last_error = e
        _init_poison_jit_registry()
        return False
#
# Privates
#


class CUDATargetOptions(TargetOptions):
    OPTIONS = {}


class CUDATarget(TargetDescriptor):
    options = CUDATargetOptions


class CUDADispatcher(object):
    targetdescr = CUDATarget

    def __init__(self, py_func, locals={}, targetoptions={}):
        assert not locals
        self.py_func = py_func
        self.targetoptions = targetoptions
        self.doc = py_func.__doc__
        self._compiled = None

    def compile(self, sig, locals={}, **targetoptions):
        assert self._compiled is None
        assert not locals
        options = self.targetoptions.copy()
        options.update(targetoptions)
        from numbapro.cudapy import jit
        kernel = jit(sig, **options)(self.py_func)
        self._compiled = kernel
        if hasattr(kernel, "_npm_context_"):
            self._npm_context_ = kernel._npm_context_

    @property
    def compiled(self):
        if self._compiled is None:
            from numbapro.cudapy import autojit
            self._compiled = autojit(self.py_func, **self.targetoptions)
        return self._compiled

    def __call__(self, *args, **kws):
        return self.compiled(*args, **kws)

    def disable_compile(self, val=True):
        """Disable the compilation of new signatures at call time.
        """
        # Do nothing
        pass

    def configure(self, *args, **kws):
        return self.compiled.configure(*args, **kws)

    def __getitem__(self, *args):
        return self.compiled.__getitem__(*args)

    def __getattr__(self, key):
        return getattr(self.compiled, key)


def CUDAPoison(*args, **kws):
    if last_error:
        raise last_error
    raise RuntimeError("CUDA devices are not available")


def _init_driver():
    from .driver import Driver
    Driver() # raises CudaSupportError


def _init_nvvm():
    from .nvvm import NVVM
    NVVM() # raises NvvmSupportError


def _init_numba_jit_registry():
    from numbapro.cudavec.vectorizers import CudaVectorize, CudaGUFuncVectorize
    target_registry['gpu'] = CUDADispatcher
    Vectorize.target_registry['gpu'] = CudaVectorize
    GUVectorize.target_registry['gpu'] = CudaGUFuncVectorize

def _init_poison_jit_registry():
    # def poison(*args, **kws):
    #     raise last_error
    target_registry['gpu'] = CUDAPoison
    Vectorize.target_registry['gpu'] = CUDAPoison
    GUVectorize.target_registry['gpu'] = CUDAPoison
