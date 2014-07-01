from __future__ import absolute_import, print_function
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from numba.cuda import jit, autojit


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
        kernel = jit(sig, **options)(self.py_func)
        self._compiled = kernel
        if hasattr(kernel, "_npm_context_"):
            self._npm_context_ = kernel._npm_context_

    @property
    def compiled(self):
        if self._compiled is None:
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

