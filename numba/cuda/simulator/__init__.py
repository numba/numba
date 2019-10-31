from __future__ import absolute_import

from .api import *
from .reduction import Reduce
from .cudadrv.devicearray import (device_array, device_array_like, pinned,
                    pinned_array, to_device, auto_device)
from .cudadrv import devicearray
from .cudadrv.devices import require_context, gpus
from .cudadrv.devices import get_context as current_context

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
