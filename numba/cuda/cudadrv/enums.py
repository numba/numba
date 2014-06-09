"""
Enum values for CUDA driver
"""
from __future__ import print_function, absolute_import, division


CUDA_SUCCESS                              = 0
CUDA_ERROR_INVALID_VALUE                  = 1
CUDA_ERROR_OUT_OF_MEMORY                  = 2
CUDA_ERROR_NOT_INITIALIZED                = 3
CUDA_ERROR_DEINITIALIZED                  = 4
CUDA_ERROR_PROFILER_DISABLED              = 5
CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6
CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7
CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8
CUDA_ERROR_NO_DEVICE                      = 100
CUDA_ERROR_INVALID_DEVICE                 = 101
CUDA_ERROR_INVALID_IMAGE                  = 200
CUDA_ERROR_INVALID_CONTEXT                = 201
CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202
CUDA_ERROR_MAP_FAILED                     = 205
CUDA_ERROR_UNMAP_FAILED                   = 206
CUDA_ERROR_ARRAY_IS_MAPPED                = 207
CUDA_ERROR_ALREADY_MAPPED                 = 208
CUDA_ERROR_NO_BINARY_FOR_GPU              = 209
CUDA_ERROR_ALREADY_ACQUIRED               = 210
CUDA_ERROR_NOT_MAPPED                     = 211
CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212
CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213
CUDA_ERROR_ECC_UNCORRECTABLE              = 214
CUDA_ERROR_UNSUPPORTED_LIMIT              = 215
CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216
CUDA_ERROR_INVALID_SOURCE                 = 300
CUDA_ERROR_FILE_NOT_FOUND                 = 301
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303
CUDA_ERROR_OPERATING_SYSTEM               = 304
CUDA_ERROR_INVALID_HANDLE                 = 400
CUDA_ERROR_NOT_FOUND                      = 500
CUDA_ERROR_NOT_READY                      = 600
CUDA_ERROR_LAUNCH_FAILED                  = 700
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701
CUDA_ERROR_LAUNCH_TIMEOUT                 = 702
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708
CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709
CUDA_ERROR_ASSERT                         = 710
CUDA_ERROR_TOO_MANY_PEERS                 = 711
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713
CUDA_ERROR_UNKNOWN                        = 999


# no preference for shared memory or L1 (default)
CU_FUNC_CACHE_PREFER_NONE    = 0x00
# prefer larger shared memory and smaller L1 cache
CU_FUNC_CACHE_PREFER_SHARED  = 0x01
# prefer larger L1 cache and smaller shared memory
CU_FUNC_CACHE_PREFER_L1      = 0x02
# prefer equal sized L1 cache and shared memory
CU_FUNC_CACHE_PREFER_EQUAL   = 0x03

# Automatic scheduling
CU_CTX_SCHED_AUTO          = 0x00
# Set spin as default scheduling
CU_CTX_SCHED_SPIN          = 0x01
# Set yield as default scheduling
CU_CTX_SCHED_YIELD         = 0x02
# Set blocking synchronization as default scheduling
CU_CTX_SCHED_BLOCKING_SYNC = 0x04

CU_CTX_SCHED_MASK          = 0x07

# Support mapped pinned allocations
CU_CTX_MAP_HOST            = 0x08
# Keep local memory allocation after launch
CU_CTX_LMEM_RESIZE_TO_MAX  = 0x10

CU_CTX_FLAGS_MASK          = 0x1f



# If set, host memory is portable between CUDA contexts.
# Flag for cuMemHostAlloc()
CU_MEMHOSTALLOC_PORTABLE = 0x01

# If set, host memory is mapped into CUDA address space and
# cuMemHostGetDevicePointer() may be called on the host pointer.
# Flag for cuMemHostAlloc()
CU_MEMHOSTALLOC_DEVICEMAP = 0x02

# If set, host memory is allocated as write-combined - fast to write,
# faster to DMA, slow to read except via SSE4 streaming load instruction
# (MOVNTDQA).
# Flag for cuMemHostAlloc()
CU_MEMHOSTALLOC_WRITECOMBINED = 0x04

# If set, host memory is portable between CUDA contexts.
# Flag for cuMemHostRegister()
CU_MEMHOSTREGISTER_PORTABLE = 0x01

# If set, host memory is mapped into CUDA address space and
# cuMemHostGetDevicePointer() may be called on the host pointer.
# Flag for cuMemHostRegister()
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02


# Default event flag
CU_EVENT_DEFAULT        = 0x0
# Event uses blocking synchronization
CU_EVENT_BLOCKING_SYNC  = 0x1
# Event will not record timing data
CU_EVENT_DISABLE_TIMING = 0x2
# Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set
CU_EVENT_INTERPROCESS   = 0x4

# The CUcontext on which a pointer was allocated or registered
CU_POINTER_ATTRIBUTE_CONTEXT = 1,
# The CUmemorytype describing the physical location of a pointer
CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
# The address at which a pointer's memory may be accessed on the device
CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
# The address at which a pointer's memory may be accessed on the host
CU_POINTER_ATTRIBUTE_HOST_POINTER = 4
# A pair of tokens for use with the nv-p2p.h Linux kernel interface
CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5

# Host memory
CU_MEMORYTYPE_HOST    = 0x01
# Device memory
CU_MEMORYTYPE_DEVICE  = 0x02
# Array memory
CU_MEMORYTYPE_ARRAY   = 0x03
# Unified device or host memory
CU_MEMORYTYPE_UNIFIED = 0x04



# Compiled device-class-specific device code
# Applicable options: none
CU_JIT_INPUT_CUBIN = 0

# PTX source code
# Applicable options: PTX compiler options
CU_JIT_INPUT_PTX = 1

# Bundle of multiple cubins and/or PTX of some device code
# Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
CU_JIT_INPUT_FATBINAR = 2

# Host object with embedded device code
# Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
CU_JIT_INPUT_OBJECT = 3

# Archive of host objects with embedded device code
# Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
CU_JIT_INPUT_LIBRARY = 4



# Max number of registers that a thread may use.
# Option type: unsigned int
# Applies to: compiler only

CU_JIT_MAX_REGISTERS = 0,


# IN: Specifies minimum number of threads per block to target compilation
# for
# OUT: Returns the number of threads the compiler actually targeted.
# This restricts the resource utilization fo the compiler (e.g. max
# registers) such that a block with the given number of threads should be
# able to launch based on register limitations. Note, this option does not
# currently take into account any other resource limitations, such as
# shared memory utilization.
# Cannot be combined with ::CU_JIT_TARGET.
# Option type: unsigned int
# Applies to: compiler only

CU_JIT_THREADS_PER_BLOCK = 1


# Overwrites the option value with the total wall clock time, in
# milliseconds, spent in the compiler and linker
# Option type: float
# Applies to: compiler and linker

CU_JIT_WALL_TIME = 2


# Pointer to a buffer in which to print any log messages
# that are informational in nature (the buffer size is specified via
# option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
# Option type: char *
# Applies to: compiler and linker

CU_JIT_INFO_LOG_BUFFER = 3


# IN: Log buffer size in bytes.  Log messages will be capped at this size
# (including null terminator)
# OUT: Amount of log buffer filled with messages
# Option type: unsigned int
# Applies to: compiler and linker

CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4


# Pointer to a buffer in which to print any log messages that
# reflect errors (the buffer size is specified via option
# ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
# Option type: char *
# Applies to: compiler and linker

CU_JIT_ERROR_LOG_BUFFER = 5


# IN: Log buffer size in bytes.  Log messages will be capped at this size
# (including null terminator)
# OUT: Amount of log buffer filled with messages
# Option type: unsigned int
# Applies to: compiler and linker

CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6


# Level of optimizations to apply to generated code (0 - 4), with 4
# being the default and highest level of optimizations.
# Option type: unsigned int
# Applies to: compiler only

CU_JIT_OPTIMIZATION_LEVEL = 7


# No option value required. Determines the target based on the current
# attached context (default)
# Option type: No option value needed
# Applies to: compiler and linker

CU_JIT_TARGET_FROM_CUCONTEXT = 8


# Target is chosen based on supplied ::CUjit_target.  Cannot be
# combined with ::CU_JIT_THREADS_PER_BLOCK.
# Option type: unsigned int for enumerated type ::CUjit_target
# Applies to: compiler and linker

CU_JIT_TARGET = 9


# Specifies choice of fallback strategy if matching cubin is not found.
# Choice is based on supplied ::CUjit_fallback.
# Option type: unsigned int for enumerated type ::CUjit_fallback
# Applies to: compiler only

CU_JIT_FALLBACK_STRATEGY = 10


# Specifies whether to create debug information in output (-g)
# (0: false, default)
# Option type: int
# Applies to: compiler and linker

CU_JIT_GENERATE_DEBUG_INFO = 11


# Generate verbose log messages (0: false, default)
# Option type: int
# Applies to: compiler and linker

CU_JIT_LOG_VERBOSE = 12


# Generate line number information (-lineinfo) (0: false, default)
# Option type: int
# Applies to: compiler only

CU_JIT_GENERATE_LINE_INFO = 13


# Specifies whether to enable caching explicitly (-dlcm)
# Choice is based on supplied ::CUjit_cacheMode_enum.
# Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum
# Applies to: compiler only

CU_JIT_CACHE_MODE = 14


# Device attributes


CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34


# CUfunction_attribute

# The maximum number of threads per block, beyond which a launch of the
# function would fail. This number depends on both the function and the
# device on which the function is currently loaded.
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0

# The size in bytes of statically-allocated shared memory required by
# this function. This does not include dynamically-allocated shared
# memory requested by the user at runtime.
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1

# The size in bytes of user-allocated constant memory required by this
# function.
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2

# The size in bytes of local memory used by each thread of this function.
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3

# The number of registers used by each thread of this function.
CU_FUNC_ATTRIBUTE_NUM_REGS = 4

# The PTX virtual architecture version for which the function was
# compiled. This value is the major PTX version * 10 + the minor PTX
# version, so a PTX version 1.3 function would return the value 13.
# Note that this may return the undefined value of 0 for cubins
# compiled prior to CUDA 3.0.
CU_FUNC_ATTRIBUTE_PTX_VERSION = 5

# The binary architecture version for which the function was compiled.
# This value is the major binary version * 10 + the minor binary version,
# so a binary version 1.3 function would return the value 13. Note that
# this will return a value of 10 for legacy cubins that do not have a
# properly-encoded binary architecture version.
CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
