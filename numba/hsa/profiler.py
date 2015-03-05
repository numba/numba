from __future__ import absolute_import, division, print_function

from numba.profiler import Profiler, DummyProfiler
from numba.config import ENABLE_HSA_PROFILER


profiler = Profiler() if ENABLE_HSA_PROFILER else DummyProfiler()

