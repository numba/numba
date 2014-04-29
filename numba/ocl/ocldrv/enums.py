"""
Enum values for OpenCL

Note that this may be platform dependent (!)
"""
from __future__ import print_function, absolute_import, division


# Error Codes
CL_SUCCESS                                    = 0

# Device Types
CL_DEVICE_TYPE_DEFAULT                        = 1<<0
CL_DEVICE_TYPE_CPU                            = 1<<1
CL_DEVICE_TYPE_GPU                            = 1<<2
CL_DEVICE_TYPE_ACCELERATOR                    = 1<<3
CL_DEVICE_TYPE_ALL                            = 0xffffffff

