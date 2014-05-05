"""
Enum values for OpenCL

Note that this may be platform dependent (!)
"""
from __future__ import print_function, absolute_import, division


# booleans
CL_TRUE                                       = 1
CL_FALSE                                      = 0

# Error Codes
CL_SUCCESS                                    = 0

# cl_coomand_queue_properties
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE        = (1 << 0)
CL_QUEUE_PROFILING_ENABLE                     = (1 << 1)

# cl_context_properties
CL_CONTEXT_PLATFORM                           = 0x1084

# Device Types
CL_DEVICE_TYPE_DEFAULT                        = (1 << 0)
CL_DEVICE_TYPE_CPU                            = (1 << 1)
CL_DEVICE_TYPE_GPU                            = (1 << 2)
CL_DEVICE_TYPE_ACCELERATOR                    = (1 << 3)
CL_DEVICE_TYPE_CUSTOM                         = (1 << 4)
CL_DEVICE_TYPE_ALL                            = 0xffffffff

# cl_platform_info
CL_PLATFORM_PROFILE                           = 0x0900
CL_PLATFORM_VERSION                           = 0x0901
CL_PLATFORM_NAME                              = 0x0902
CL_PLATFORM_VENDOR                            = 0x0903
CL_PLATFORM_EXTENSIONS                        = 0x0904

# cl_device_info
CL_DEVICE_TYPE                                = 0x1000
CL_DEVICE_VENDOR_ID                           = 0x1001
CL_DEVICE_MAX_COMPUTE_UNITS                   = 0x1002
CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS            = 0x1003
CL_DEVICE_MAX_WORK_GROUP_SIZE                 = 0x1004
CL_DEVICE_MAX_WORK_ITEM_SIZES                 = 0x1005
CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR         = 0x1006
CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT        = 0x1007
CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT          = 0x1008
CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG         = 0x1009
CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT        = 0x100A
CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE       = 0x100B
CL_DEVICE_MAX_CLOCK_FREQUENCY                 = 0x100C
CL_DEVICE_ADDRESS_BITS                        = 0x100D
CL_DEVICE_MAX_READ_IMAGE_ARGS                 = 0x100E
CL_DEVICE_MAX_WRITE_IMAGE_ARGS                = 0x100F
CL_DEVICE_MAX_MEM_ALLOC_SIZE                  = 0x1010
CL_DEVICE_IMAGE2D_MAX_WIDTH                   = 0x1011
CL_DEVICE_IMAGE2D_MAX_HEIGHT                  = 0x1012
CL_DEVICE_IMAGE3D_MAX_WIDTH                   = 0x1013
CL_DEVICE_IMAGE3D_MAX_HEIGHT                  = 0x1014
CL_DEVICE_IMAGE3D_MAX_DEPTH                   = 0x1015
CL_DEVICE_IMAGE_SUPPORT                       = 0x1016
CL_DEVICE_MAX_PARAMETER_SIZE                  = 0x1017
CL_DEVICE_MAX_SAMPLERS                        = 0x1018
CL_DEVICE_MEM_BASE_ADDR_ALIGN                 = 0x1019
CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE            = 0x101A
CL_DEVICE_SINGLE_FP_CONFIG                    = 0x101B
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE               = 0x101C
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE           = 0x101D
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE               = 0x101E
CL_DEVICE_GLOBAL_MEM_SIZE                     = 0x101F
CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE            = 0x1020
CL_DEVICE_MAX_CONSTANT_ARGS                   = 0x1021
CL_DEVICE_LOCAL_MEM_TYPE                      = 0x1022
CL_DEVICE_LOCAL_MEM_SIZE                      = 0x1023
CL_DEVICE_ERROR_CORRECTION_SUPPORT            = 0x1024
CL_DEVICE_PROFILING_TIMER_RESOLUTION          = 0x1025
CL_DEVICE_ENDIAN_LITTLE                       = 0x1026
CL_DEVICE_AVAILABLE                           = 0x1027
CL_DEVICE_COMPILER_AVAILABLE                  = 0x1028
CL_DEVICE_EXECUTION_CAPABILITIES              = 0x1029
CL_DEVICE_QUEUE_PROPERTIES                    = 0x102A
CL_DEVICE_NAME                                = 0x102B
CL_DEVICE_VENDOR                              = 0x102C
CL_DRIVER_VERSION                             = 0x102D
CL_DEVICE_PROFILE                             = 0x102E
CL_DEVICE_VERSION                             = 0x102F
CL_DEVICE_EXTENSIONS                          = 0x1030
CL_DEVICE_PLATFORM                            = 0x1031
CL_DEVICE_DOUBLE_FP_CONFIG                    = 0x1032
CL_DEVICE_HALF_FP_CONFIG                      = 0x1033      
CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF         = 0x1034
CL_DEVICE_HOST_UNIFIED_MEMORY                 = 0x1035
CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR            = 0x1036
CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT           = 0x1037
CL_DEVICE_NATIVE_VECTOR_WIDTH_INT             = 0x1038
CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG            = 0x1039
CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT           = 0x103A
CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE          = 0x103B
CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF            = 0x103C
CL_DEVICE_OPENCL_C_VERSION                    = 0x103D
CL_DEVICE_LINKER_AVAILABLE                    = 0x103E
CL_DEVICE_BUILT_IN_KERNELS                    = 0x103F
CL_DEVICE_IMAGE_MAX_BUFFER_SIZE               = 0x1040
CL_DEVICE_IMAGE_MAX_ARRAY_SIZE                = 0x1041
CL_DEVICE_PARENT_DEVICE                       = 0x1042
CL_DEVICE_PARTITION_MAX_SUB_DEVICES           = 0x1043
CL_DEVICE_PARTITION_PROPERTIES                = 0x1044
CL_DEVICE_PARTITION_AFFINITY_DOMAIN           = 0x1045
CL_DEVICE_PARTITION_TYPE                      = 0x1046
CL_DEVICE_REFERENCE_COUNT                     = 0x1047
CL_DEVICE_PREFERRED_INTEROP_USER_SYNC         = 0x1048
CL_DEVICE_PRINTF_BUFFER_SIZE                  = 0x1049
CL_DEVICE_IMAGE_PITCH_ALIGNMENT               = 0x104A
CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT        = 0x104B

#cl_mem_flags
CL_MEM_READ_WRITE                             = (1 << 0)
CL_MEM_WRITE_ONLY                             = (1 << 1)
CL_MEM_READ_ONLY                              = (1 << 2)
CL_MEM_USE_HOST_PTR                           = (1 << 3)
CL_MEM_ALLOC_HOST_PTR                         = (1 << 4)
CL_MEM_COPY_HOST_PTR                          = (1 << 5)
CL_MEM_HOST_WRITE_ONLY                        = (1 << 7)
CL_MEM_HOST_READ_ONLY                         = (1 << 8)
CL_MEM_HOST_NO_ACCESS                         = (1 << 9)

# cl_kernel_work_group_info
CL_KERNEL_WORK_GROUP_SIZE                     = 0x11B0
CL_KERNEL_COMPILE_WORK_GROUP_SIZE             = 0x11B1
CL_KERNEL_LOCAL_MEM_SIZE                      = 0x11B2
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE  = 0x11B3
CL_KERNEL_PRIVATE_MEM_SIZE                    = 0x11B4
CL_KERNEL_GLOBAL_WORK_SIZE                    = 0x11B5

# error codes
CL_SUCCESS                                    =  0   
CL_DEVICE_NOT_FOUND                           = -1  
CL_DEVICE_NOT_AVAILABLE                       = -2  
CL_COMPILER_NOT_AVAILABLE                     = -3  
CL_MEM_OBJECT_ALLOCATION_FAILURE              = -4  
CL_OUT_OF_RESOURCES                           = -5  
CL_OUT_OF_HOST_MEMORY                         = -6  
CL_PROFILING_INFO_NOT_AVAILABLE               = -7  
CL_MEM_COPY_OVERLAP                           = -8  
CL_IMAGE_FORMAT_MISMATCH                      = -9  
CL_IMAGE_FORMAT_NOT_SUPPORTED                 = -10 
CL_BUILD_PROGRAM_FAILURE                      = -11 
CL_MAP_FAILURE                                = -12 
CL_MISALIGNED_SUB_BUFFER_OFFSET               = -13 
CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST  = -14
CL_COMPILE_PROGRAM_FAILURE                    = -15 
CL_LINKER_NOT_AVAILABLE                       = -16 
CL_LINK_PROGRAM_FAILURE                       = -17 
CL_DEVICE_PARTITION_FAILED                    = -18 
CL_KERNEL_ARG_INFO_NOT_AVAILABLE              = -19 
