from __future__ import absolute_import

from .api import *
from .array import (devicearray, device_array, device_array_like, pinned,
                    pinned_array, to_device)
from .reduction import Reduce
reduce = Reduce

# Ensure that any user code attempting to import cudadrv etc. gets the
# simulator's version and not the real version if the simulator is enabled.
from numba import config
if config.ENABLE_CUDASIM:
    import sys
    from . import cudadrv
    sys.modules['numba.cuda.cudadrv'] = cudadrv
    sys.modules['numba.cuda.cudadrv.devicearray'] = cudadrv.devicearray
    sys.modules['numba.cuda.cudadrv.devices'] = cudadrv.devices
    sys.modules['numba.cuda.cudadrv.driver'] = cudadrv.driver
    sys.modules['numba.cuda.cudadrv.drvapi'] = cudadrv.drvapi
    sys.modules['numba.cuda.cudadrv.nvvm'] = cudadrv.nvvm

    from . import compiler
    sys.modules['numba.cuda.compiler'] = compiler
