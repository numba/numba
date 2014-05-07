"""OpenCL Driver

- Driver API binding
- Device array implementation

Based on cuda driver
"""
from __future__ import print_function, absolute_import, division

from .enums import *
from . import types
from .driver import driver, opencl_strerror, OpenCLSupportError, OpenCLDriverError, OpenCLAPIError
