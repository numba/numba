"""Enum values for HSA from the HSA extension header

Note that Python namespacing could be used to avoid the C-like
prefixing, but we choose to keep the same names as found in the C
enums, in order to match the documentation.
"""

import ctypes

#
# Agent attributes
#
# Enums of the type hsa_amd_agent_info_t
#

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

# Internal driver node identifier. The type of this attribute is uint32_t.
HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = 0xA004

# Max number of watch points on memory address ranges to generate exception
# events when the watched addresses are accessed.
HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS = 0xA005

#
# Region attributes
#
# Enums of the type hsa_amd_region_info_t
#

# Determine if host can access the region. The type of this attribute is bool.
HSA_AMD_REGION_INFO_HOST_ACCESSIBLE = 0xA000

# Base address of the region in flat address space.
HSA_AMD_REGION_INFO_BASE = 0xA001

#
# Coherency attributes of a fine grained region
#
# Enums of the type hsa_amd_coherency_type_t
#

# Coherent region.
HSA_AMD_COHERENCY_TYPE_COHERENT = 0

# Non coherent region.
HSA_AMD_COHERENCY_TYPE_NONCOHERENT = 1

