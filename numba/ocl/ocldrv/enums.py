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

# cl_platform_info
CL_PLATFORM_PROFILE                           = 0x0900
CL_PLATFORM_VERSION                           = 0x0901
CL_PLATFORM_NAME                              = 0x0902
CL_PLATFORM_VENDOR                            = 0x0903
CL_PLATFORM_EXTENSIONS                        = 0x0904
