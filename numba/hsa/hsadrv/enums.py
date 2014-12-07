"""Enum values for HSA

Note that Python namespacing could be used to avoid the C-like
prefixing, but we choose to keep the same names as found in the C
enums, in order to match the documentation.
"""

import ctypes

HSA_LARGE_MODEL = ctypes.sizeof(ctypes.c_void_p) == 8

# hsa_status_t
HSA_STATUS_SUCCESS                            = 0
HSA_STATUS_INFO_BREAK                         = 0x1
HSA_EXT_STATUS_INFO_ALREADY_INITIALIZED       = 0x4000
HSA_EXT_STATUS_INFO_UNRECOGNIZED_OPTIONS      = 0x4001
HSA_STATUS_ERROR                              = 0x10000
HSA_STATUS_ERROR_INVALID_ARGUMENT             = 0x10001
HSA_STATUS_ERROR_INVALID_QUEUE_CREATION       = 0x10002
HSA_STATUS_ERROR_INVALID_ALLOCATION           = 0x10003
HSA_STATUS_ERROR_INVALID_AGENT                = 0x10004
HSA_STATUS_ERROR_INVALID_REGION               = 0x10005
HSA_STATUS_ERROR_INVALID_SIGNAL               = 0x10006
HSA_STATUS_ERROR_INVALID_QUEUE                = 0x10007
HSA_STATUS_ERROR_OUT_OF_RESOURCES             = 0x10008
HSA_STATUS_ERROR_INVALID_PACKET_FORMAT        = 0x10009
HSA_STATUS_ERROR_RESOURCE_FREE                = 0x1000A
HSA_STATUS_ERROR_NOT_INITIALIZED              = 0x1000B
HSA_STATUS_ERROR_REFCOUNT_OVERFLOW            = 0x1000C
HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH       = 0x14000
HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED = 0x14001
HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED   = 0x14002

# hsa_packet_type_t
HSA_PACKET_TYPE_ALWAYS_RESERVED               = 0
HSA_PACKET_TYPE_INVALID                       = 1
HSA_PACKET_TYPE_DISPATCH                      = 2
HSA_PACKET_TYPE_BARRIER                       = 3
HSA_PACKET_TYPE_AGENT_DISPATCH                = 4

# hsa_queue_type_t
HSA_QUEUE_TYPE_MULTI                          = 0
HSA_QUEUE_TYPE_SINGLE                         = 1

# hsa_queue_feature_t
HSA_QUEUE_FEATURE_DISPATCH                    = 1
HSA_QUEUE_FEATURE_AGENT_DISPATCH              = 2

# hsa_fence_scope_t
HSA_FENCE_SCOPE_NONE                          = 0
HSA_FENCE_SCOPE_COMPONENT                     = 1
HSA_FENCE_SCOPE_SYSTEM                        = 2

# hsa_wait_expectancy_t
HSA_WAIT_EXPECTANCY_SHORT                     = 0
HSA_WAIT_EXPECTANCY_LONG                      = 1
HSA_WAIT_EXPECTANCY_UNKNOWN                   = 2

# hsa_signal_condition_t
HSA_EQ                                        = 0
HSA_NE                                        = 1
HSA_LT                                        = 2
HSA_GTE                                       = 3

# hsa_dim_t
HSA_DIM_X                                     = 0
HSA_DIM_Y                                     = 1
HSA_DIM_Z                                     = 2

# hsa_extension_t
HSA_EXT_START                                 = 0
HSA_EXT_FINALIZER                             = HSA_EXT_START
HSA_EXT_LINKER                                = 1
HSA_EXT_IMAGES                                = 2
HSA_SVEXT_START                               = 10000

# hsa_agent_feature_t
HSA_AGENT_FEATURE_DISPATCH                    = 1
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

# hsa_agent_info_t
HSA_AGENT_INFO_NAME                           = 0
HSA_AGENT_INFO_VENDOR_NAME                    = 1
HSA_AGENT_INFO_FEATURE                        = 2
HSA_AGENT_INFO_WAVEFRONT_SIZE                 = 3
HSA_AGENT_INFO_WORKGROUP_MAX_DIM              = 4
HSA_AGENT_INFO_WORKGROUP_MAX_SIZE             = 5
HSA_AGENT_INFO_GRID_MAX_DIM                   = 6
HSA_AGENT_INFO_GRID_MAX_SIZE                  = 7
HSA_AGENT_INFO_FBARRIER_MAX_SIZE              = 8
HSA_AGENT_INFO_QUEUES_MAX                     = 9
HSA_AGENT_INFO_QUEUE_MAX_SIZE                 = 10
HSA_AGENT_INFO_QUEUE_TYPE                     = 11
HSA_AGENT_INFO_NODE                           = 12
HSA_AGENT_INFO_DEVICE                         = 13
HSA_AGENT_INFO_CACHE_SIZE                     = 14
HSA_EXT_AGENT_INFO_IMAGE1D_MAX_DIM            = 15
HSA_EXT_AGENT_INFO_IMAGE2D_MAX_DIM            = 16
HSA_EXT_AGENT_INFO_IMAGE3D_MAX_DIM            = 17
HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_SIZE       = 18
HSA_EXT_AGENT_INFO_IMAGE_RD_MAX               = 19
HSA_EXT_AGENT_INFO_IMAGE_RDWR_MAX             = 20
HSA_EXT_AGENT_INFO_SAMPLER_MAX                = 21

# hsa_segment_t
HSA_SEGMENT_GLOBAL                            = 0
HSA_SEGMENT_PRIVATE                           = 1
HSA_SEGMENT_GROUP                             = 2
HSA_SEGMENT_KERNARG                           = 3
HSA_SEGMENT_READONLY                          = 4
HSA_SEGMENT_SPILL                             = 5
HSA_SEGMENT_ARG                               = 6

# hsa_region_flag_t
HSA_REGION_FLAG_KERNARG                       = 1
HSA_REGION_FLAG_CACHED_L1                     = 2
HSA_REGION_FLAG_CACHED_L2                     = 4
HSA_REGION_FLAG_CACHED_L3                     = 8
HSA_REGION_FLAG_CACHED_L4                     = 16

# hsa_region_info_t
HSA_REGION_INFO_BASE                          = 0
HSA_REGION_INFO_SIZE                          = 1
HSA_REGION_INFO_AGENT                         = 2
HSA_REGION_INFO_FLAGS                         = 3
HSA_REGION_INFO_SEGMENTS                      = 4
HSA_REGION_INFO_ALLOC_MAX_SIZE                = 5
HSA_REGION_INFO_ALLOC_GRANULE                 = 6
HSA_REGION_INFO_ALLOC_ALIGNMENT               = 7
HSA_REGION_INFO_BANDWIDTH                     = 8
HSA_REGION_INFO_NODE                          = 9
HSA_REGION_INFO_COUNT                         = 10

# hsa_powertwo_t
HSA_POWERTWO_1                                = 0
HSA_POWERTWO_2                                = 1
HSA_POWERTWO_4                                = 2
HSA_POWERTWO_8                                = 3
HSA_POWERTWO_16                               = 4
HSA_POWERTWO_32                               = 5
HSA_POWERTWO_64                               = 6
HSA_POWERTWO_128                              = 7
HSA_POWERTWO_256                              = 8


# hsa_system_info_t
HSA_SYSTEM_INFO_VERSION_MAJOR                 = 0
HSA_SYSTEM_INFO_VERSION_MINOR                 = 1
HSA_SYSTEM_INFO_TIMESTAMP                     = 2
HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY           = 3
HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT               = 4

# hsa_ext_exception_kind_t
HSA_EXT_EXCEPTION_INVALID_OPERATION           = 1
HSA_EXT_EXCEPTION_DIVIDE_BY_ZERO              = 2
HSA_EXT_EXCEPTION_OVERFLOW                    = 4
HSA_EXT_EXCEPTION_UNDERFLOW                   = 8
HSA_EXT_EXCEPTION_INEXACT                     = 16

# hsa_ext_control_directive_present_t
HSA_EXT_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS       = 0
HSA_EXT_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS      = 1
HSA_EXT_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE        = 2
HSA_EXT_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE            = 4
HSA_EXT_CONTROL_DIRECTIVE_MAX_FLOAT_WORKGROUP_SIZE      = 8
HSA_EXT_CONTROL_DIRECTIVE_REQUESTED_WORKGROUPS_PER_CU   = 16
HSA_EXT_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE            = 32
HSA_EXT_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE       = 64
HSA_EXT_CONTROL_DIRECTIVE_REQUIRED_DIM                  = 128
HSA_EXT_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS = 256

# hsa_ext_brig_profile_t
HSA_EXT_BRIG_PROFILE_BASE                     = 0
HSA_EXT_BRIG_PROFILE_FULL                     = 1

# hsa_ext_brig_machine_model_t
HSA_EXT_BRIG_MACHINE_SMALL                    = 0
HSA_EXT_BRIG_MACHINE_LARGE                    = 1

# hsa_ext_brig_section_id_t
HSA_EXT_BRIG_SECTION_DATA                     = 0
HSA_EXT_BRIG_SECTION_CODE                     = 1
HSA_EXT_BRIG_SECTION_OPERAND                  = 2

# hsa_ext_code_kind_t
HSA_EXT_CODE_NONE                             = 0
HSA_EXT_CODE_KERNEL                           = 1
HSA_EXT_CODE_INDIRECT_FUNCTION                = 2
HSA_EXT_CODE_RUNTIME_FIRST                    = 0x40000000
HSA_EXT_CODE_RUNTIME_LAST                     = 0x7fffffff
HSA_EXT_CODE_VENDOR_FIRST                     = 0x80000000
HSA_EXT_CODE_VENDOR_LAST                      = 0xffffffff

# hsa_ext_program_call_convention_id_t
HSA_EXT_PROGRAM_CALL_CONVENTION_FINALIZER_DETERMINED = -1
