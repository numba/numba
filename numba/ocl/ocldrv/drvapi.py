"""
This file contains ctypes signatures for the OpenCL functions. This
is represented as the API_PROTOTYPES dictionary that maps the API
function name to its signature in the form of (return type, arg_types ... ).

This is not exhaustive, and prototypes will be added on an "as needed"
basis.

Error handling: Unlike CUDA that is highly regular, error codes
maybe in different places for different functions. Typically this
will be:
- return value, for functions that would otherwise return void.
- last argument, by reference, for functions where there is a 
  clear return value (for example, "create" functions return the
  created object and have as last argument an optional pointer to
  cl_int to save any error code).
- none at all, when it cannot fail (does this happen?)

The last argument in the prototype will be used to identify the error
code argument. It will typically be either 0 (return value),
-1 (last argument) or None (no error code)
"""

from __future__ import print_function, absolute_import, division

from ctypes import c_void_p, c_size_t, c_char_p, POINTER as ptr

from .types import *


API_PROTOTYPES = {
    'clGetPlatformIDs': (cl_int, cl_uint, ptr(cl_platform_id), ptr(cl_uint), 0),
    'clGetPlatformInfo': (cl_int, cl_platform_id, cl_platform_info, c_size_t, c_void_p, ptr(c_size_t), 0),
    'clGetDeviceIDs': (cl_int, cl_platform_id, cl_device_type, cl_uint, ptr(cl_device_id), ptr(cl_uint), 0),
    'clGetDeviceInfo': (cl_int, cl_device_id, cl_device_info, c_size_t, c_void_p, ptr(c_size_t), 0),
    'clCreateContext': (cl_context, ptr(cl_context_properties), cl_uint, ptr(cl_device_id), c_void_p, ptr(cl_int), -1),
    'clRetainContext': (cl_int, cl_context, 0),
    'clReleaseContext': (cl_int, cl_context, 0),
    'clCreateCommandQueue': (cl_command_queue, cl_context, cl_device_id, cl_command_queue_properties, ptr(cl_int), -1),
    'clRetainCommandQueue': (cl_int, cl_context, 0),
    'clReleaseCommandQueue': (cl_int, cl_command_queue, 0),
    'clCreateBuffer': (cl_mem, cl_context, cl_mem_flags, c_size_t, c_void_p, ptr(cl_int), -1),
    'clRetainMemObject': (cl_int, cl_mem, 0),
    'clReleaseMemObject': (cl_int, cl_mem, 0),
    'clCreateProgramWithSource': (cl_program, cl_context, cl_uint, ptr(c_char_p), ptr(c_size_t), ptr(cl_int), -1),
    'clBuildProgram': (cl_int, cl_program, cl_uint, ptr(cl_device_id), c_char_p, c_void_p, c_void_p, 0),
    'clRetainProgram': (cl_int, cl_program, 0), 
    'clReleaseProgram': (cl_int, cl_program, 0), 
    'clCreateKernel': (cl_kernel, cl_program, c_char_p, ptr(cl_int), -1),
    'clRetainKernel': (cl_int, cl_kernel, 0),
    'clReleaseKernel': (cl_int, cl_kernel, 0),
    'clSetKernelArg': (cl_int, cl_kernel, cl_uint, c_size_t, c_void_p, 0),
    'clGetKernelWorkGroupInfo': (cl_int, cl_kernel, cl_device_id, cl_kernel_work_group_info, c_size_t, c_void_p, ptr(c_size_t), 0),
    'clEnqueueTask': (cl_int, cl_command_queue, cl_kernel, cl_uint, ptr(cl_event), ptr(cl_event), 0),
    'clEnqueueNDRangeKernel': (cl_int, cl_command_queue, cl_kernel, cl_uint, ptr(c_size_t), ptr(c_size_t), ptr(c_size_t), cl_uint, ptr(cl_event), ptr(cl_event), 0),
    'clEnqueueReadBuffer': (cl_int, cl_command_queue, cl_mem, cl_bool, c_size_t, c_size_t, c_void_p, cl_uint, ptr(cl_event), ptr(cl_event), 0),
    'clEnqueueWriteBuffer': (cl_int, cl_command_queue, cl_mem, cl_bool, c_size_t, c_size_t, c_void_p, cl_uint, ptr(cl_event), ptr(cl_event), 0),
    'clFlush': (cl_int, cl_command_queue, 0),
    'clFinish': (cl_int, cl_command_queue, 0)
}
