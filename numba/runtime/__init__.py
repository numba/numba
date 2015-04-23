from . import nrt
from numba.utils import finalize as _finalize

# initialize
rtsys = nrt.Runtime()

# install finalizer
_finalize(rtsys, nrt.Runtime.shutdown)
