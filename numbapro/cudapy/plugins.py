from __future__ import print_function, absolute_import, division
from functools import reduce
import operator
from numbapro.cudadrv.autotune import AutoTuner
from numba.cuda.compiler import CUDAKernel


# Extend CUDAKernel class

def autotune(self):
    has_autotune = (hasattr(self, '_autotune') and
                    self._autotune.dynsmem == self.sharedmem)
    if has_autotune:
        return self._autotune
    else:
        cinfo = self._func.get_info()
        at = AutoTuner.parse(self.entry_name, cinfo,
                             cc=self.device.compute_capability)
        if at is None:
            raise RuntimeError('driver does not report compiliation info')
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

