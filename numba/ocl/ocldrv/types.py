"""
type definitions for opencl types (mapping to ctypes).

This is based on:
https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/scalarDataTypes.html

And to some point it will be based on 
"""

from __future__ import print_function, absolute_import, division

import ctypes

cl_char = ctypes.c_int8 # signed two's complement 8-bit integer
cl_uchar = ctypes.c_uint8 # unsigned 8-bit integer
cl_short = ctypes.c_int16 # signed two's complement 16-bit integer
cl_ushort = ctypes.c_uint16 # unsigend 16-bit integer
cl_int = ctypes.c_int32 # signed two's complement 32-bit integer
cl_uint = ctypes.c_uint32 # unsigned 32-bit integer
cl_long = ctypes.c_int64 # signed two's complement 64-bit integer
cl_ulong = ctypes.c_uint64 # unsigned 64-bit integer


cl_platform_id = ctypes.c_void_p
cl_device_id = ctypes.c_void_p
cl_context = ctypes.c_void_p
cl_command_queue = ctypes.c_void_p
cl_mem = ctypes.c_void_p
cl_program = ctypes.c_void_p
cl_kernel = ctypes.c_void_p
cl_event = ctypes.c_void_p
cl_sampler = ctypes.c_void_p
cl_mem = ctypes.c_void_p

# The ones below may need to be tweaked per platform (looked up on include files)
cl_bitfield = cl_ulong
cl_command_queue_properties = cl_bitfield
cl_device_type = cl_bitfield
cl_mem_flags = cl_bitfield
cl_platform_info = cl_uint
cl_device_info = cl_uint
cl_device_fp_config = cl_bitfield
cl_device_exec_capabilities = cl_bitfield
cl_device_mem_cache_type = cl_uint
cl_device_local_mem_type = cl_uint
cl_bool = cl_uint # this probably can change from platform to platform
