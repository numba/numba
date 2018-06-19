"""Enum values for HSA from the HSA extension header

Note that Python namespacing could be used to avoid the C-like
prefixing, but we choose to keep the same names as found in the C
enums, in order to match the documentation.
"""

# These enums are a direct translation of those found in:
# hsa_ext_amd.h from the ROCR-Runtime. For example:
# https://github.com/RadeonOpenCompute/ROCR-Runtime/blob/master/src/inc/hsa_ext_amd.h
# Comments relating to the values are largely wholesale copied.

import ctypes


#------------------------------------------------------------------------------
#
# Anonymous enum expressing that a memory pool is invalid
#
HSA_STATUS_ERROR_INVALID_MEMORY_POOL = 40
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Agent attributes
#
# Enums of the type hsa_amd_agent_info_t

# Chip identifier. The type of this attribute is uint32_t.
HSA_AMD_AGENT_INFO_CHIP_ID = 0xA000

# Size of a cacheline in bytes. The type of this attribute is uint32_t.
HSA_AMD_AGENT_INFO_CACHELINE_SIZE = 0xA001

# The number of compute unit available in the agent. The type of this
# attribute is uint32_t.
HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = 0xA002

# The maximum clock frequency of the agent in MHz. The type of this
# attribute is uint32_t.
HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY = 0xA003

# Internay driver node identifier. The type of this attribute is uint32_t.
HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = 0xA004

# Max number of watch points on memory address ranges to generate exception
# events when the watched addresses are accessed.
HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS = 0xA005
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Region attributes
#
# Enums of the type hsa_amd_region_info_t

# Determine if host can access the region. The type of this attribute is bool.
HSA_AMD_REGION_INFO_HOST_ACCESSIBLE = 0xA000

# Base address of the region in flat address space.
HSA_AMD_REGION_INFO_BASE = 0xA001

# Memory Interface width, the return value type is uint32_t.
# This attribute is deprecated. Use HSA_AMD_AGENT_INFO_MEMORY_WIDTH.
HSA_AMD_REGION_INFO_BUS_WIDTH = 0xA002

# Max Memory Clock, the return value type is uint32_t.
# This attribute is deprecated. Use HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY.
HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY = 0xA003
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Coherency attributes of a fine grained region
#
# Enums of the type hsa_amd_coherency_type_t

# Coherent region.
HSA_AMD_COHERENCY_TYPE_COHERENT = 0

# Non coherent region.
HSA_AMD_COHERENCY_TYPE_NONCOHERENT = 1
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Memory segments associated with a memory pool.
#
# Enums of the type hsa_amd_segment_t

# Global segment. Used to hold data that is shared by all agents.
HSA_AMD_SEGMENT_GLOBAL = 0

# Read-only segment. Used to hold data that remains constant during the
# execution of a kernel.
HSA_AMD_SEGMENT_READONLY = 1

# Private segment. Used to hold data that is local to a single work-item.
HSA_AMD_SEGMENT_PRIVATE = 2

# Group segment. Used to hold data that is shared by the work-items of a
# work-group.
HSA_AMD_SEGMENT_GROUP = 3
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Memory pool global flags.
#
# Enums of the type hsa_amd_memory_pool_global_flag_t.

# The application can use allocations in the memory pool to store kernel
# arguments, and provide the values for the kernarg segment of
# a kernel dispatch.
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT = 1

# Updates to memory in this pool conform to HSA memory consistency model.
# If this flag is set, then HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED
# must not be set.
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 2

# Writes to memory in this pool can be performed by a single agent at a time.
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 4
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Memory pool features flags.
#
# Enums of the type hsa_amd_memory_pool_info_t.

# Segment where the memory pool resides. The type of this attribute is
# hsa_amd_segment_t.
HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0

# Flag mask. The value of this attribute is undefined if the value of
# HSA_AMD_MEMORY_POOL_INFO_SEGMENT is not HSA_AMD_SEGMENT_GLOBAL. The type
# of this attribute is uint32_t, a bit-field of
# hsa_amd_memory_pool_global_flag_t values.
HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1

# Size of this pool, in bytes. The type of this attribute is size_t.
HSA_AMD_MEMORY_POOL_INFO_SIZE = 2

# Indicates whether memory in this pool can be allocated using
# hsa_amd_memory_pool_allocate. The type of this attribute is bool.
# The value of this flag is always false for memory pools in the group and
# private segments.
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5

# Allocation granularity of buffers allocated by hsa_amd_memory_pool_allocate
# in this memory pool. The size of a buffer allocated in this pool is a
# multiple of the value of this attribute. The value of this attribute is
# only defined if HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED is true for
# this pool. The type of this attribute is size_t.
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = 6

# Alignment of buffers allocated by hsa_amd_memory_pool_allocate in this
# pool. The value of this attribute is only defined if
# HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED is true for this pool, and
# must be a power of 2. The type of this attribute is size_t.
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT = 7

# This memory_pool can be made directly accessible by all the agents in the
# system (hsa_amd_agent_memory_pool_get_info returns
# HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT for all agents). The type of
# this attribute is bool.
HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL = 15
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Type of accesses to a memory pool from a given agent.
#
# Enums of the type hsa_amd_memory_pool_access_t

# The agent cannot directly access any buffer in the memory pool.
HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED = 0

# The agent can directly access a buffer located in the pool; the application
# does not need to invoke hsa_amd_agents_allow_access.
HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT = 1

# The agent can directly access a buffer located in the pool, but only if the
# application has previously requested access to that buffer using
# hsa_amd_agents_allow_access.
HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT = 2
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Properties of the relationship between an agent a memory pool.
#
# Enums of the type hsa_amd_link_info_type_t

# Hyper-transport bus type.
HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT = 0

# QPI bus type.
HSA_AMD_LINK_INFO_TYPE_QPI = 1

# PCIe bus type.
HSA_AMD_LINK_INFO_TYPE_PCIE = 2

# Infiniband bus type.
HSA_AMD_LINK_INFO_TYPE_INFINBAND = 3
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Access to buffers located in the memory pool. The type of this attribute
# is hsa_amd_memory_pool_access_t.
#
# Enums of type hsa_amd_agent_memory_pool_info_t.

# An agent can always directly access buffers currently located in a memory
# pool that is associated (the memory_pool is one of the values returned by
# hsa_amd_agent_iterate_memory_pools on the agent) with that agent. If the
# buffer is currently located in a memory pool that is not associated with
# the agent, and the value returned by this function for the given
# combination of agent and memory pool is not
# HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED, the application still needs to
# invoke hsa_amd_agents_allow_access in order to gain direct access to the
# buffer.

# If the given agent can directly access buffers the pool, the result is not
# HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED. If the memory pool is associated
# with the agent, or it is of fined-grained type, the result must not be
# HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED. If the memory pool is not
# associated with the agent, and does not reside in the global segment, the
# result must be HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED.
HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = 0

# Number of links to hop when accessing the memory pool from the specified
# agent. The type of this attribute is uint32_t.
HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS = 1

# Details of each link hop when accessing the memory pool starting from the
# specified agent. The type of this attribute is an array size of
# HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS with each element containing
# hsa_amd_memory_pool_link_info_t.
HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO = 2
#------------------------------------------------------------------------------


