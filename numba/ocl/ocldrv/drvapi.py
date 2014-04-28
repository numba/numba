"""
This file contains ctypes signatures for the OpenCL functions. This
is represented as the API_PROTOTYPES dictionary that maps the API
function name to its signature in the form of (return type, arg_types ... ).

This is not exhaustive, and prototypes will be added on an "as needed"
basis.
"""

from __future__ import print_function, absolute_import, division

from ctypes import c_void_p, c_size_t, c_char_p, POINTER as ptr

from .types import *



cl_int = ct.int32

API_PROTOTYPES = {
    'clGetPlatformIDs': (cl_int, cluint, ptr(cl_platform_id), ptr(cl_uint)),
    'clGetDeviceIDs': (cl_int, cl_platform_id, cl_device_type, cl_uint, ptr(cl_device_id), ptr(cl_uint)),
    'clCreateContext': (cl_context, c_void_p, cl_uint, ptr(cl_device_id), c_void_p, ptr(cl_int)),
    'clRetainContext': (cl_int, cl_context),
    'clReleaseContext': (cl_int, cl_context),
    'clCreateCommandQueue': (cl_command_queue, cl_context, cl_device_id, cl_command_queue_properties, ptr(cl_int)),
    'clRetainCommandQueue': (cl_int, cl_context),
    'clReleaseCommandQueue': (cl_int, cl_command_queue),
    'clCreateBuffer': (cl_mem, cl_context, cl_mem_flags, c_size_t, c_void_p, ptr(cl_int)),
    'clRetainMemObject': (cl_int, cl_mem),
    'clReleaseMemObject': (cl_int, cl_mem),
    'clCreateProgramWithSource': (cl_program, cl_context, cl_uint, ptr(c_char_p), ptr(c_size_t), ptr(cl_int)),
    'clBuildProgram': (cl_int, cl_program, cl_uint, ptr(cl_device_id), c_char_p, c_void_p, c_void_p),
    'clRetainProgram': (cl_int, cl_program), 
    'clReleaseProgram': (cl_int, cl_program), 
    'clCreateKernel': (cl_kernel, cl_program, c_char_p, ptr(cl_int)),
    'clRetainKernel': (cl_int, cl_kernel),
    'clReleaseKernel': (cl_int, cl_kernel),
    'clSetKernelArg': (cl_int, cl_kernel, cl_uint, c_size_t, c_void_p),
    'clEnqueueTask': (cl_int, cl_command_queue, cl_kernel, cl_uint, ptr(cl_event), ptr(cl_event)),
    'clEnqueueReadBuffer': (cl_int, cl_command_queue, cl_mem, cl_bool, c_size_t, c_size_t, c_void_p, cl_uint, ptr(cl_event), ptr(cl_event)),
    'clFlush': (cl_int, cl_command_queue),
    'clFinish': (cl_int, cl_command_queue)
}
