from typing import Any, Final, Literal

import numpy as np

# `array_exprs` is `del`ed at runtime but required by stubtest
from numba.np.ufunc import array_exprs  # noqa: F401
from numba.np.ufunc.decorators import GUVectorize as GUVectorize
from numba.np.ufunc.decorators import Vectorize as Vectorize
from numba.np.ufunc.decorators import guvectorize as guvectorize
from numba.np.ufunc.decorators import vectorize as vectorize

# originally defined in `numba.np.ufunc._internal`
PyUFunc_Zero: Final = 0
PyUFunc_One: Final = 1
PyUFunc_None: Final = -1
PyUFunc_ReorderableNone: Final = -2

# originally defined in `numba.np.ufunc.parallel`
def threading_layer() -> Literal["tbb", "omp", "workqueue"]: ...
def set_num_threads(n: int | np.integer[Any]) -> None: ...
def get_num_threads() -> int: ...
def get_thread_id() -> int: ...
def set_parallel_chunksize(n: int | np.integer[Any]) -> int: ...
def get_parallel_chunksize() -> int: ...
