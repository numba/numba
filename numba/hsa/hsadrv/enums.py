"""Enum values for HSA

Note that Python namespacing could be used to avoid the C-like
prefixing, but we choose to keep the same names as found in the C
enums, in order to match the documentation.
"""

import ctypes

HSA_LARGE_MODEL = ctypes.sizeof(ctypes.c_void_p) == 8

# hsa_status_t

# The function has been executed successfully.
HSA_STATUS_SUCCESS = 0x0
# A traversal over a list of elements has been interrupted by the
# application before completing.
HSA_STATUS_INFO_BREAK = 0x1
# A generic error has occurred.
HSA_STATUS_ERROR = 0x1000
# One of the actual arguments does not meet a precondition stated in the
# documentation of the corresponding formal argument.
HSA_STATUS_ERROR_INVALID_ARGUMENT = 0x1001
# The requested queue creation is not valid.
HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = 0x1002
# The requested allocation is not valid.
HSA_STATUS_ERROR_INVALID_ALLOCATION = 0x1003
# The agent is invalid.
HSA_STATUS_ERROR_INVALID_AGENT = 0x1004
# The memory region is invalid.
HSA_STATUS_ERROR_INVALID_REGION = 0x1005
# The signal is invalid.
HSA_STATUS_ERROR_INVALID_SIGNAL = 0x1006
# The queue is invalid.
HSA_STATUS_ERROR_INVALID_QUEUE = 0x1007
# The HSA runtime failed to allocate the necessary resources. This error
# may also occur when the HSA runtime needs to spawn threads or create
# internal OS-specific events.
HSA_STATUS_ERROR_OUT_OF_RESOURCES = 0x1008
# The AQL packet is malformed.
HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = 0x1009
# An error has been detected while releasing a resource.
HSA_STATUS_ERROR_RESOURCE_FREE = 0x100A
# An API other than ::hsa_init has been invoked while the reference count
# of the HSA runtime is 0.
HSA_STATUS_ERROR_NOT_INITIALIZED = 0x100B
# The maximum reference count for the object has been reached.
HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = 0x100C
# The arguments passed to a functions are not compatible.
HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = 0x100D
# The index is invalid.\
HSA_STATUS_ERROR_INVALID_INDEX = 0x100E
# The instruction set architecture is invalid.
HSA_STATUS_ERROR_INVALID_ISA = 0x100F,
# The instruction set architecture name is invalid.
HSA_STATUS_ERROR_INVALID_ISA_NAME = 0x1017
# The code object is invalid.
HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 0x1010
# The executable is invalid.
HSA_STATUS_ERROR_INVALID_EXECUTABLE = 0x1011
# The executable is frozen.
HSA_STATUS_ERROR_FROZEN_EXECUTABLE = 0x1012
# There is no symbol with the given name.
HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = 0x1013
# The variable is already defined.
HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = 0x1014
# The variable is undefined.
HSA_STATUS_ERROR_VARIABLE_UNDEFINED = 0x1015
# An HSAIL operation resulted on a hardware exception.
HSA_STATUS_ERROR_EXCEPTION = 0x1016

# hsa_packet_type_t
HSA_PACKET_TYPE_VENDOR_SPECIFIC               = 0
# The packet has been processed in the past, but has not been reassigned to
# the packet processor. A packet processor must not process a packet of this
# type. All queues support this packet type.
HSA_PACKET_TYPE_INVALID                       = 1
# Packet used by agents for dispatching jobs to kernel agents. Not all
# queues support packets of this type (see ::hsa_queue_feature_t).
HSA_PACKET_TYPE_KERNEL_DISPATCH               = 2
# Packet used by agents to delay processing of subsequent packets, and to
# express complex dependencies between multiple packets. All queues support
# this packet type.
HSA_PACKET_TYPE_BARRIER_AND                   = 3
# Packet used by agents for dispatching jobs to agents.  Not all
# queues support packets of this type (see ::hsa_queue_feature_t).
HSA_PACKET_TYPE_AGENT_DISPATCH                = 4
# Packet used by agents to delay processing of subsequent packets, and to
# express complex dependencies between multiple packets. All queues support
# this packet type.
HSA_PACKET_TYPE_BARRIER_OR                    = 5

# hsa_queue_type_t
HSA_QUEUE_TYPE_MULTI                          = 0
HSA_QUEUE_TYPE_SINGLE                         = 1

# hsa_queue_feature_t
HSA_QUEUE_FEATURE_KERNEL_DISPATCH                    = 1
HSA_QUEUE_FEATURE_AGENT_DISPATCH              = 2

# hsa_fence_scope_t
HSA_FENCE_SCOPE_NONE                          = 0
HSA_FENCE_SCOPE_AGENT                         = 1
HSA_FENCE_SCOPE_SYSTEM                        = 2

# hsa_wait_state_t
# The application thread may be rescheduled while waiting on the signal.
HSA_WAIT_STATE_BLOCKED = 0
# The application thread stays active while waiting on a signal.
HSA_WAIT_STATE_ACTIVE = 1

# hsa_signal_condition_t
HSA_SIGNAL_CONDITION_EQ                                        = 0
HSA_SIGNAL_CONDITION_NE                                        = 1
HSA_SIGNAL_CONDITION_LT                                        = 2
HSA_SIGNAL_CONDITION_GTE                                       = 3

# # hsa_dim_t
# HSA_DIM_X                                     = 0
# HSA_DIM_Y                                     = 1
# HSA_DIM_Z                                     = 2

# hsa_extension_t
HSA_EXTENSION_FINALIZER = 0
HSA_EXTENSION_IMAGES = 1
HSA_EXTENSION_AMD_PROFILER = 2

# hsa_agent_feature_t
HSA_AGENT_FEATURE_KERNEL_DISPATCH             = 1
HSA_AGENT_FEATURE_AGENT_DISPATCH              = 2

# hsa_device_type_t
HSA_DEVICE_TYPE_CPU                           = 0
HSA_DEVICE_TYPE_GPU                           = 1
HSA_DEVICE_TYPE_DSP                           = 2

# hsa_system_info_t
HSA_SYSTEM_INFO_VERSION_MAJOR                 = 0
HSA_SYSTEM_INFO_VERSION_MINOR                 = 1
HSA_SYSTEM_INFO_TIMESTAMP                     = 2
HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY           = 3
HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT               = 4
HSA_SYSTEM_INFO_ENDIANNESS                    = 5
HSA_SYSTEM_INFO_MACHINE_MODEL                 = 6
HSA_SYSTEM_INFO_EXTENSIONS                    = 7

# hsa_agent_info_t

# Agent name. The type of this attribute is a NUL-terminated char[64]. If
# the name of the agent uses less than 63 characters, the rest of the
# array must be filled with NULs.
HSA_AGENT_INFO_NAME = 0
# Name of vendor. The type of this attribute is a NUL-terminated char[64]. If
# the name of the vendor uses less than 63 characters, the rest of the array
# must be filled with NULs.
HSA_AGENT_INFO_VENDOR_NAME = 1
# Agent capability. The type of this attribute is ::hsa_agent_feature_t.
HSA_AGENT_INFO_FEATURE = 2
# Machine model supported by the agent. The type of this attribute is
# ::hsa_machine_model_t.
HSA_AGENT_INFO_MACHINE_MODEL = 3
# Profile supported by the agent. The type of this attribute is
# ::hsa_profile_t.
HSA_AGENT_INFO_PROFILE = 4
# Default floating-point rounding mode. The type of this attribute is
# ::hsa_default_float_rounding_mode_t, but the value
# ::HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT is not allowed.
HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5
# Default floating-point rounding modes supported by the agent in the Base
# profile. The type of this attribute is a mask of
# ::hsa_default_float_rounding_mode_t. The default floating-point rounding
# mode (::HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE) bit must not be set.
HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 23
# Flag indicating that the f16 HSAIL operation is at least as fast as the
# f32 operation in the current agent. The value of this attribute is
# undefined if the agent is not a kernel agent. The type of this
# attribute is bool.
HSA_AGENT_INFO_FAST_F16_OPERATION = 24
# Number of work-items in a wavefront. Must be a power of 2 in the range
# [1,256]. The value of this attribute is undefined if the agent is not
# a kernel agent. The type of this attribute is uint32_t.
HSA_AGENT_INFO_WAVEFRONT_SIZE = 6
# Maximum number of work-items of each dimension of a work-group.  Each
# maximum must be greater than 0. No maximum can exceed the value of
# ::HSA_AGENT_INFO_WORKGROUP_MAX_SIZE. The value of this attribute is
# undefined if the agent is not a kernel agent. The type of this
# attribute is uint16_t[3].
HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7
# Maximum total number of work-items in a work-group. The value of this
# attribute is undefined if the agent is not a kernel agent. The type
# of this attribute is uint32_t.
HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = 8
# Maximum number of work-items of each dimension of a grid. Each maximum must
# be greater than 0, and must not be smaller than the corresponding value in
# ::HSA_AGENT_INFO_WORKGROUP_MAX_DIM. No maximum can exceed the value of
# ::HSA_AGENT_INFO_GRID_MAX_SIZE. The value of this attribute is undefined if
# the agent is not a kernel agent. The type of this attribute is
# ::hsa_dim3_t.
HSA_AGENT_INFO_GRID_MAX_DIM = 9
# Maximum total number of work-items in a grid. The value of this attribute
# is undefined if the agent is not a kernel agent. The type of this
# attribute is uint32_t.
HSA_AGENT_INFO_GRID_MAX_SIZE = 10
# Maximum number of fbarriers per work-group. Must be at least 32. The value
# of this attribute is undefined if the agent is not a kernel agent. The
# type of this attribute is uint32_t.
HSA_AGENT_INFO_FBARRIER_MAX_SIZE = 11
# Maximum number of queues that can be active (created but not destroyed) at
# one time in the agent. The type of this attribute is uint32_t.
HSA_AGENT_INFO_QUEUES_MAX = 12
# Minimum number of packets that a queue created in the agent
# can hold. Must be a power of 2 greater than 0. Must not exceed
# the value of ::HSA_AGENT_INFO_QUEUE_MAX_SIZE. The type of this
# attribute is uint32_t.
HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13
# Maximum number of packets that a queue created in the agent can
# hold. Must be a power of 2 greater than 0. The type of this attribute
# is uint32_t.
HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14
# Type of a queue created in the agent. The type of this attribute is
# ::hsa_queue_type_t.
HSA_AGENT_INFO_QUEUE_TYPE = 15
# Identifier of the NUMA node associated with the agent. The type of this
# attribute is uint32_t.
HSA_AGENT_INFO_NODE = 16
# Type of hardware device associated with the agent. The type of this
# attribute is ::hsa_device_type_t.
HSA_AGENT_INFO_DEVICE = 17
# Array of data cache sizes (L1..L4). Each size is expressed in bytes. A size
# of 0 for a particular level indicates that there is no cache information
# for that level. The type of this attribute is uint32_t[4].
HSA_AGENT_INFO_CACHE_SIZE = 18
# Instruction set architecture of the agent. The type of this attribute
# is ::hsa_isa_t.
HSA_AGENT_INFO_ISA = 19
# Bit-mask indicating which extensions are supported by the agent. An
# extension with an ID of @p i is supported if the bit at position @p i is
# set. The type of this attribute is uint8_t[128].
HSA_AGENT_INFO_EXTENSIONS = 20
# Major version of the HSA runtime specification supported by the
# agent. The type of this attribute is uint16_t.
HSA_AGENT_INFO_VERSION_MAJOR = 21
# Minor version of the HSA runtime specification supported by the
# agent. The type of this attribute is uint16_t.
HSA_AGENT_INFO_VERSION_MINOR = 22

# hsa_region_segment_t
# Global segment. Used to hold data that is shared by all agents.
HSA_REGION_SEGMENT_GLOBAL = 0
# Read-only segment. Used to hold data that remains constant during the
# execution of a kernel.
HSA_REGION_SEGMENT_READONLY = 1
# Private segment. Used to hold data that is local to a single work-item.
HSA_REGION_SEGMENT_PRIVATE = 2
# Group segment. Used to hold data that is shared by the work-items of a
# work-group.
HSA_REGION_SEGMENT_GROUP = 3

# hsa_region_global_flag_t
# The application can use memory in the region to store kernel arguments, and
# provide the values for the kernarg segment of a kernel dispatch. If this
# flag is set, then ::HSA_REGION_GLOBAL_FLAG_FINE_GRAINED must be set.
HSA_REGION_GLOBAL_FLAG_KERNARG = 1
# Updates to memory in this region are immediately visible to all the
# agents under the terms of the HSA memory model. If this
# flag is set, then ::HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED must not be set.
HSA_REGION_GLOBAL_FLAG_FINE_GRAINED = 2
# Updates to memory in this region can be performed by a single agent at
# a time. If a different agent in the system is allowed to access the
# region, the application must explicitely invoke ::hsa_memory_assign_agent
# in order to transfer ownership to that agent for a particular buffer.
HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED = 4

# hsa_region_info_t

# Segment where memory in the region can be used. The type of this
# attribute is ::hsa_region_segment_t.
HSA_REGION_INFO_SEGMENT = 0
# Flag mask. The value of this attribute is undefined if the value of
# ::HSA_REGION_INFO_SEGMENT is not ::HSA_REGION_SEGMENT_GLOBAL. The type of
# this attribute is uint32_t, a bit-field of ::hsa_region_global_flag_t
# values.
HSA_REGION_INFO_GLOBAL_FLAGS = 1
# Size of this region, in bytes. The type of this attribute is size_t.
HSA_REGION_INFO_SIZE = 2
# Maximum allocation size in this region, in bytes. Must not exceed the value
# of ::HSA_REGION_INFO_SIZE. The type of this attribute is size_t.
#
# If the region is in the global or readonly segments, this is the maximum
# size that the application can pass to ::hsa_memory_allocate. If the region
# is in the group segment, this is the maximum size (per work-group) that can
# be requested for a given kernel dispatch. If the region is in the private
# segment, this is the maximum size (per work-item) that can be request for a
# specific kernel dispatch.
HSA_REGION_INFO_ALLOC_MAX_SIZE = 4
# Indicates whether memory in this region can be allocated using
# ::hsa_memory_allocate. The type of this attribute is bool.
#
# The value of this flag is always false for regions in the group and private
# segments.
HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED = 5
# Allocation granularity of buffers allocated by ::hsa_memory_allocate in
# this region. The size of a buffer allocated in this region is a multiple of
# the value of this attribute. The value of this attribute is only defined if
# ::HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED is true for this region. The type
# of this attribute is size_t.
HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE = 6
# Alignment of buffers allocated by ::hsa_memory_allocate in this region. The
# value of this attribute is only defined if
# ::HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED is true for this region, and must
# be a power of 2. The type of this attribute is size_t.
HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT = 7


# hsa_profile_t
HSA_PROFILE_BASE                     = 0
HSA_PROFILE_FULL                     = 1

# hsa_machine_model_t
HSA_MACHINE_MODEL_SMALL = 0
HSA_MACHINE_MODEL_LARGE = 1


# hsa_executable_symbol_info_t


# The kind of the symbol. The type of this attribute is ::hsa_symbol_kind_t.
HSA_EXECUTABLE_SYMBOL_INFO_TYPE = 0
# The length of the symbol name. The type of this attribute is uint32_t.
HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = 1
# The name of the symbol. The type of this attribute is character array with
# the length equal to the value of ::HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH
# attribute
HSA_EXECUTABLE_SYMBOL_INFO_NAME = 2
# The length of the module name to which this symbol belongs if this symbol
# has module linkage, otherwise 0 is returned. The type of this attribute is
# uint32_t.
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH = 3
# The module name to which this symbol belongs if this symbol has module
# linkage, otherwise empty string is returned. The type of this attribute is
# character array with the length equal to the value of
# ::HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH attribute.
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME = 4
# Agent associated with this symbol. If the symbol is a variable, the
# value of this attribute is only defined if
# ::HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION is
# ::HSA_VARIABLE_ALLOCATION_AGENT. The type of this attribute is hsa_agent_t.
HSA_EXECUTABLE_SYMBOL_INFO_AGENT = 20
# The address of the variable. The value of this attribute is undefined if
# the symbol is not a variable. The type of this attribute is uint64_t.
# If executable's state is ::HSA_EXECUTABLE_STATE_UNFROZEN, then 0 is
# returned.
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21
# The linkage kind of the symbol. The type of this attribute is
# ::hsa_symbol_linkage_t.
HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE = 5
# Indicates whether the symbol corresponds to a definition. The type of this
# attribute is bool.
HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION = 17
# The allocation kind of the variable. The value of this attribute is
# undefined if the symbol is not a variable.  The type of this attribute is
# ::hsa_variable_allocation_t.
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION = 6
# The segment kind of the variable. The value of this attribute is undefined
# if the symbol is not a variable. The type of this attribute is
# ::hsa_variable_segment_t.
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT = 7
# Alignment of the variable. The value of this attribute is undefined if
# the symbol is not a variable. The type of this attribute is uint32_t.
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT = 8
# Size of the variable. The value of this attribute is undefined if
# the symbol is not a variable. The type of this attribute is uint32_t.
#
# A value of 0 is returned if the variable is an external variable and has an
# unknown dimension.
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = 9
# Indicates whether the variable is constant. The value of this attribute is
# undefined if the symbol is not a variable. The type of this attribute is
# bool.
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST = 10

# Kernel object handle, used in the kernel dispatch packet. The value of this
# attribute is undefined if the symbol is not a kernel. The type of this
# attribute is uint64_t.
#
# If the state of the executable is ::HSA_EXECUTABLE_STATE_UNFROZEN, then 0
# is returned.
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = 22
# Size of kernarg segment memory that is required to hold the values of the
# kernel arguments, in bytes. The value of this attribute is undefined if the
# symbol is not a kernel. The type of this attribute is uint32_t.
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11
# Alignment (in bytes) of the buffer used to pass arguments to the kernel,
# which is the maximum of 16 and the maximum alignment of any of the kernel
# arguments. The value of this attribute is undefined if the symbol is not a
# kernel. The type of this attribute is uint32_t.
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = 12
# Size of static group segment memory required by the kernel (per
# work-group), in bytes. The value of this attribute is undefined
# if the symbol is not a kernel. The type of this attribute is uint32_t.
#
# The reported amount does not include any dynamically allocated group
# segment memory that may be requested by the application when a kernel is
# dispatched.
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13
# Size of static private, spill, and arg segment memory required by
# this kernel (per work-item), in bytes. The value of this attribute is
# undefined if the symbol is not a kernel. The type of this attribute is
# uint32_t.
#
# If the value of ::HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK is
# true, the kernel may use more private memory than the reported value, and
# the application must add the dynamic call stack usage to @a
# private_segment_size when populating a kernel dispatch packet.
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14
# Dynamic callstack flag. The value of this attribute is undefined if the
# symbol is not a kernel. The type of this attribute is bool.
#
# If this flag is set (the value is true), the kernel uses a dynamically
# sized call stack. This can happen if recursive calls, calls to indirect
# functions, or the HSAIL alloca instruction are present in the kernel.
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = 15
# Indirect function object handle. The value of this attribute is undefined
# if the symbol is not an indirect function, or the associated agent does
# not support the Full Profile. The type of this attribute depends on the
# machine model: if machine model is small, then the type is uint32_t, if
# machine model is large, then the type is uint64_t.
#
# If the state of the executable is ::HSA_EXECUTABLE_STATE_UNFROZEN, then 0
# is returned.
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT = 23
# Call convention of the indirect function. The value of this attribute is
# undefined if the symbol is not an indirect function, or the associated
# agent does not support the Full Profile. The type of this attribute is
# uint32_t.
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = 16


# hsa_default_float_rounding_mode_t

# Use a default floating-point rounding mode specified elsewhere.
HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT = 0
# Operations that specify the default floating-point mode are rounded to zero
# by default.
HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO = 1
# Operations that specify the default floating-point mode are rounded to the
# nearest representable number and that ties should be broken by selecting
# the value with an even least significant bit.
HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR = 2

# hsa_code_object_type_t
HSA_CODE_OBJECT_TYPE_PROGRAM = 0


# hsa_executable_state_t

# Executable state, which allows the user to load code objects and define
# external variables. Variable addresses, kernel code handles, and
# indirect function code handles are not available in query operations until
# the executable is frozen (zero always returned).

HSA_EXECUTABLE_STATE_UNFROZEN = 0

# Executable state, which allows the user to query variable addresses,
# kernel code handles, and indirect function code handles using query
# operation. Loading new code objects, as well as defining external variables
# is not allowed in this state.

HSA_EXECUTABLE_STATE_FROZEN = 1


# hsa_kernel_dispatch_packet_setup_t
HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = 0



# hsa_packet_header_t
HSA_PACKET_HEADER_TYPE = 0
HSA_PACKET_HEADER_BARRIER = 8
HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE = 9
HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE = 11

