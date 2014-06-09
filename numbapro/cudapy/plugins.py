from __future__ import print_function, absolute_import, division
from functools import reduce
import operator
from numbapro.cudadrv.autotune import AutoTuner
from numba.cuda.compiler import CUDAKernel, CUDAKernelBase, AutoJitCUDAKernel
from numba.cuda.descriptor import CUDATargetDesc
from . import printimpl

# Extend target features

CUDATargetDesc.targetctx.insert_func_defn(printimpl.registry.functions)


# Extend CUDAKernel class

class CompilationInfoUnavailable(RuntimeError):
    pass


def autotune(self):
    has_autotune = hasattr(self, '_autotune')
    if has_autotune and self._autotune.dynsmem == self.sharedmem:
        return self._autotune
    else:
        # Get CUDA Function
        cufunc = self._func.get()
        at = AutoTuner(info=cufunc.attrs, cc=cufunc.device.compute_capability)
        if at is None:
            raise CompilationInfoUnavailable(
                "driver does not report compiliation info")
        self._autotune = at
        return self._autotune


def occupancy(self):
    """calculate the theoretical occupancy of the kernel given the
    configuration.
    """
    thread_per_block = reduce(operator.mul, self.blockdim, 1)
    return self.autotune.closest(thread_per_block)


CUDAKernel.autotune = property(autotune)
CUDAKernel.occupancy = property(occupancy)


# Extend CUDAKernelBase class

def forall(self, ntasks, tpb=128, stream=0, sharedmem=0):
    """
    Returns a configured kernel for 1D kernel of given number of tasks
    ``ntasks``.

    This assumes that:
    - the kernel 1-to-1 maps global thread id ``cuda.grid(1)`` to tasks.
    - the kernel must check if the thread id is valid.

    """
    return ForAll(self, ntasks, tpb=tpb, stream=stream, sharedmem=sharedmem)


def _compute_thread_per_block(kernel, tpb):
    if tpb != 0:
        return tpb

    else:
        try:
            tpb = kernel.autotune.best(tpb)
        except CompilationInfoUnavailable:
            tpb = 128

        return tpb


class ForAll(object):
    def __init__(self, kernel, ntasks, tpb, stream, sharedmem):
        self.kernel = kernel
        self.ntasks = ntasks
        self.thread_per_block = tpb
        self.stream = stream
        self.sharedmem = sharedmem

    def __call__(self, *args):
        if isinstance(self.kernel, AutoJitCUDAKernel):
            kernel = self.kernel.specialize(*args)
        else:
            kernel = self.kernel

        tpb = _compute_thread_per_block(kernel, self.thread_per_block)
        tpbm1 = tpb - 1
        blkct = (self.ntasks + tpbm1) // tpb

        return kernel.configure(blkct, tpb, stream=self.stream,
                                sharedmem=self.sharedmem)(*args)


CUDAKernelBase.forall = forall
