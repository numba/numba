import ctypes
import warnings

from numba.core import utils

from numba.roc.hsadrv import enums
from .error import HsaApiError, HsaWarning

_PTR = ctypes.POINTER

# This deals with types which are defined as
# typedef struct { uint64_t handle;};
handle_struct = ctypes.c_uint64

#------------------------------------------------------------------------------
# HSA types from hsa.h, ordered as per header file

hsa_status_t = ctypes.c_int # enum
class hsa_dim3_t(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_uint32),
        ('y', ctypes.c_uint32),
        ('z', ctypes.c_uint32)
        ]
hsa_access_permission_t  = ctypes.c_int # enum
hsa_endianness_t  = ctypes.c_int # enum
hsa_machine_model_t  = ctypes.c_int # enum
hsa_profile_t  = ctypes.c_int # enum
hsa_system_info_t  = ctypes.c_int # enum
hsa_extension_t = ctypes.c_int # enum
hsa_agent_t = handle_struct
hsa_agent_feature_t = ctypes.c_int # enum
hsa_device_type_t = ctypes.c_int # enum
hsa_default_float_rounding_mode_t = ctypes.c_int # enum
hsa_agent_info_t = ctypes.c_int # enum
hsa_exception_policy_t = ctypes.c_int # enum
hsa_signal_t = handle_struct
hsa_signal_value_t = ctypes.c_uint64 if enums.HSA_LARGE_MODEL else ctypes.c_uint32
hsa_signal_condition_t = ctypes.c_int # enum
hsa_wait_state_t = ctypes.c_int # enum
hsa_region_t = handle_struct
hsa_queue_type_t = ctypes.c_int # enum
hsa_queue_feature_t = ctypes.c_int # enum
class hsa_queue_t(ctypes.Structure):
    """In theory, this should be aligned to 64 bytes. In any case, allocation
    of this structure is done by the hsa library"""
    _fields_ = [
        ('type', hsa_queue_type_t),
        ('features', ctypes.c_uint32),
        ('base_address', ctypes.c_void_p),  # if LARGE MODEL
        ('doorbell_signal', hsa_signal_t),
        ('size', ctypes.c_uint32),
        ('reserved1', ctypes.c_uint32),
        ('id', ctypes.c_uint32),
        ]
hsa_packet_type_t = ctypes.c_int # enum
hsa_fence_scope_t = ctypes.c_int # enum
hsa_packet_header_t = ctypes.c_int # enum
hsa_packet_header_width_t = ctypes.c_int # enum
hsa_kernel_dispatch_packet_setup_t = ctypes.c_int # enum
hsa_kernel_dispatch_packet_setup_width_t = ctypes.c_int # enum
class hsa_kernel_dispatch_packet_t(ctypes.Structure):
    _fields_ = [
        ('header', ctypes.c_uint16),
        ('setup', ctypes.c_uint16),
        ('workgroup_size_x', ctypes.c_uint16),
        ('workgroup_size_y', ctypes.c_uint16),
        ('workgroup_size_z', ctypes.c_uint16),
        ('reserved0', ctypes.c_uint16), # Must be zero
        ('grid_size_x', ctypes.c_uint32),
        ('grid_size_y', ctypes.c_uint32),
        ('grid_size_z', ctypes.c_uint32),
        ('private_segment_size', ctypes.c_uint32),
        ('group_segment_size', ctypes.c_uint32),
        ('kernel_object', ctypes.c_uint64),
        # NOTE: Small model not dealt with properly...!
        # ifdef HSA_LARGE_MODEL
        ('kernarg_address', ctypes.c_uint64),
        # SMALL Machine has a reserved uint32
        ('reserved2', ctypes.c_uint64), # Must be zero
        ('completion_signal', hsa_signal_t),
        ]
class hsa_agent_dispatch_packet_t(ctypes.Structure):
        """This should be aligned to HSA_PACKET_ALIGN_BYTES (64)"""
        _fields_ = [
            ('header', ctypes.c_uint16),
            ('type', ctypes.c_uint16),
            ('reserved0', ctypes.c_uint32),
            # NOTE: Small model not dealt with properly...!
            ('return_address', ctypes.c_void_p),
            ('arg', ctypes.c_uint64 * 4),
            ('reserved2', ctypes.c_uint64),
            ('completion_signal', hsa_signal_t),
        ]
class hsa_barrier_and_packet_t(ctypes.Structure):
    _fields_ = [
        ('header', ctypes.c_uint16),
        ('reserved0', ctypes.c_uint16),
        ('reserved1', ctypes.c_uint32),
        ('dep_signal0', hsa_signal_t),
        ('dep_signal1', hsa_signal_t),
        ('dep_signal2', hsa_signal_t),
        ('dep_signal3', hsa_signal_t),
        ('dep_signal4', hsa_signal_t),
        ('reserved2', ctypes.c_uint64),
        ('completion_signal', hsa_signal_t),
        ]

hsa_barrier_or_packet_t = hsa_barrier_and_packet_t

hsa_region_segment_t = ctypes.c_int # enum
hsa_region_global_flag_t = ctypes.c_int # enum
hsa_region_info_t = ctypes.c_int # enum
hsa_symbol_kind_t = ctypes.c_int # enum
hsa_variable_allocation_t = ctypes.c_int # enum
hsa_symbol_linkage_t = ctypes.c_int # enum
hsa_variable_segment_t = ctypes.c_int # enum
hsa_isa_t = handle_struct
hsa_isa_info_t = ctypes.c_int # enum
hsa_code_object_t = handle_struct
hsa_callback_data_t = handle_struct
hsa_code_object_type_t = ctypes.c_int # enum
hsa_code_object_info_t = ctypes.c_int # enum
hsa_code_symbol_t = handle_struct
hsa_code_symbol_info_t = ctypes.c_int # enum
hsa_executable_t = handle_struct
hsa_executable_state_t = ctypes.c_int # enum
hsa_executable_info_t = ctypes.c_int # enum
hsa_executable_symbol_t = handle_struct
hsa_executable_symbol_info_t = ctypes.c_int # enum
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# HSA types from Brig.h, ordered as per header file
# NOTE: not all of the definitions are needed
BrigVersion32_t = ctypes.c_uint32
MODULE_IDENTIFICATION_LENGTH=8
class BrigModuleHeader(ctypes.Structure):
    _fields_ = [
        ('identification', ctypes.c_char*MODULE_IDENTIFICATION_LENGTH),
        ('brigMajor', BrigVersion32_t),
        ('brigMinor', BrigVersion32_t),
        ('byteCount', ctypes.c_uint64),
        ('hash', ctypes.c_uint8*64),
        ('reserved',  ctypes.c_uint32),
        ('sectionCount', ctypes.c_uint32),
        ('sectionIndex', ctypes.c_uint64),
    ]

BrigModule_t = _PTR(BrigModuleHeader)

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# HSA types from hsa_ext_amd.h, ordered as per header file
hsa_amd_agent_info_t = ctypes.c_int # enum
hsa_amd_region_info_t = ctypes.c_int # enum
hsa_amd_coherency_type_t = ctypes.c_int # enum
class hsa_amd_profiling_dispatch_time_t(ctypes.Structure):
    _fields_ = [
        ('start', ctypes.c_uint64),
        ('end', ctypes.c_uint64),
        ]

# typedef bool (*hsa_amd_signal_handler)(hsa_signal_value_t value, void* arg);
hsa_amd_signal_handler = _PTR(
    ctypes.CFUNCTYPE(ctypes.c_bool,
                     hsa_signal_value_t,
                     ctypes.c_void_p)
    )

hsa_amd_segment_t = ctypes.c_int # enum
hsa_amd_memory_pool_t = handle_struct
hsa_amd_memory_pool_global_flag_t = ctypes.c_int # enum
hsa_amd_memory_pool_info_t = ctypes.c_int # enum
hsa_amd_memory_pool_access_t = ctypes.c_int # enum
hsa_amd_link_info_type_t = ctypes.c_int # enum
hsa_amd_memory_pool_link_info_t = ctypes.c_int # enum
hsa_amd_agent_memory_pool_info_t = ctypes.c_int # enum
class hsa_amd_image_descriptor_t(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('deviceID', ctypes.c_uint32),
        ('data', ctypes.c_uint32*1),
        ]
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# HSA types from hsa_ext_finalize.h, ordered as per header file
hsa_ext_module_t = BrigModule_t

hsa_ext_program_t = handle_struct
hsa_ext_program_info_t = ctypes.c_int # enum
hsa_ext_finalizer_call_convention_t = ctypes.c_int # enum
class hsa_ext_control_directives_t(ctypes.Structure):
    _fields_ = [
        ('control_directives_mask', ctypes.c_uint64),
        ('break_exceptions_mask', ctypes.c_uint16),
        ('detect_exceptions_mask', ctypes.c_uint16),
        ('max_dynamic_group_size', ctypes.c_uint32),
        ('max_flat_grid_size', ctypes.c_uint64),
        ('max_flat_workgroup_size', ctypes.c_uint32),
        ('reserved1', ctypes.c_uint32),
        ('required_grid_size', ctypes.c_uint64*3),
        ('required_workgroup_size', hsa_dim3_t),
        ('required_dim', ctypes.c_uint8),
        ('reserved2', ctypes.c_uint8*75),
    ]

# function pointers, that are used in the
# "hsa_ext_finalizer_1_00_pfn_t" struct of pointers
HSA_EXT_PROGRAM_CREATE_FPTR = ctypes.CFUNCTYPE(
        hsa_status_t, # return value
        hsa_machine_model_t, # machine_model
        hsa_profile_t, # profile
        hsa_default_float_rounding_mode_t, # default_float_rounding_mode
        ctypes.c_char_p, # options
        _PTR(hsa_ext_program_t)) # program

HSA_EXT_PROGRAM_DESTROY_FPTR  = ctypes.CFUNCTYPE(
        hsa_status_t, # return value
        hsa_ext_program_t) # program

HSA_EXT_PROGRAM_ADD_MODULE_FPTR = ctypes.CFUNCTYPE(
        hsa_status_t, # return value
        hsa_ext_program_t, # program
        hsa_ext_module_t) # module

HSA_EXT_PROGRAM_ITERATE_MODULES_CALLBACK_FUNC = ctypes.CFUNCTYPE(
        hsa_status_t, # return
        hsa_ext_program_t, # program
        hsa_ext_module_t, # module
        ctypes.c_void_p) # data

HSA_EXT_PROGRAM_ITERATE_MODULES_FPTR = ctypes.CFUNCTYPE(
        hsa_status_t, # return value
        hsa_ext_program_t, # program
        HSA_EXT_PROGRAM_ITERATE_MODULES_CALLBACK_FUNC, # callback
        ctypes.c_void_p) # data

HSA_EXT_PROGRAM_GET_INFO_FPTR = ctypes.CFUNCTYPE(
        hsa_status_t, # return value
        hsa_ext_program_t, # program
        hsa_ext_program_info_t, # attribute
        ctypes.c_void_p) # value

HSA_EXT_PROGRAM_FINALIZE_FPTR = ctypes.CFUNCTYPE(
        hsa_status_t, # return value
        hsa_ext_program_t, # program
        hsa_isa_t, # isa
        ctypes.c_int32, # call_convention
        hsa_ext_control_directives_t, # control_directives
        ctypes.c_char_p, #options
        hsa_code_object_type_t, #code_object_type
        _PTR(hsa_code_object_t)) # code_object

# this struct holds function pointers
class hsa_ext_finalizer_1_00_pfn_t(ctypes.Structure):
    _fields_ = [
               ('hsa_ext_program_create', HSA_EXT_PROGRAM_CREATE_FPTR),
               ('hsa_ext_program_destroy', HSA_EXT_PROGRAM_DESTROY_FPTR),
               ('hsa_ext_program_add_module', HSA_EXT_PROGRAM_ADD_MODULE_FPTR),
               ('hsa_ext_program_iterate_modules',
                   HSA_EXT_PROGRAM_ITERATE_MODULES_FPTR),
               ('hsa_ext_program_get_info', HSA_EXT_PROGRAM_GET_INFO_FPTR),
               ('hsa_ext_program_finalize', HSA_EXT_PROGRAM_FINALIZE_FPTR)
    ]

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# HSA types from hsa_ext_image.h (NOTE: support incomplete)

hsa_ext_image_t = handle_struct
hsa_ext_image_geometry_t = ctypes.c_int # enum
hsa_ext_image_channel_type_t = ctypes.c_int # enum
hsa_ext_image_channel_order_t = ctypes.c_int # enum

class hsa_ext_image_format_t(ctypes.Structure):
    _fields_ = [
        ("channel_type", hsa_ext_image_channel_type_t),
        ("channel_order", hsa_ext_image_channel_order_t)
    ]

class hsa_ext_image_descriptor_t(ctypes.Structure):
    _fields_ = [
        ("geometry", hsa_ext_image_geometry_t),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("depth", ctypes.c_size_t),
        ("array_size", ctypes.c_size_t),
        ("format", hsa_ext_image_format_t)
    ]

hsa_ext_image_capability_t = ctypes.c_int # enum

class hsa_ext_image_data_info_t(ctypes.Structure):
    _fields_ = [
             ("size", ctypes.c_size_t),
             ("alignment", ctypes.c_size_t),
             ]

class hsa_ext_image_region_t(ctypes.Structure):
    _fields_ = [
             ("offset", hsa_dim3_t),
             ("offset", hsa_dim3_t),
    ]

hsa_ext_sampler_t = handle_struct
hsa_ext_sampler_addressing_mode_t = ctypes.c_int # enum
hsa_ext_sampler_coordinate_mode_t = ctypes.c_int # enum
hsa_ext_sampler_filter_mode_t = ctypes.c_int # enum

class hsa_ext_sampler_descriptor_t(ctypes.Structure):
    _fields_ = [
        ("coordinate_mode", hsa_ext_sampler_coordinate_mode_t),
        ("filter_mode", hsa_ext_sampler_filter_mode_t),
        ("address_mode", hsa_ext_sampler_addressing_mode_t)
    ]

#NOTE: Not implemented yet: hsa_ext_images_1_00_pfn_t
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# callbacks that have no related typedef in the hsa include files

HSA_ITER_AGENT_CALLBACK_FUNC = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_agent_t, # agent
    ctypes.py_object) # this is a c_void_p used to wrap a python object

HSA_QUEUE_CALLBACK_FUNC = ctypes.CFUNCTYPE(
    None,  # return value
    hsa_status_t,
    _PTR(hsa_queue_t),
    ctypes.py_object) # this is a c_void_p used to wrap a python object

HSA_AGENT_ITERATE_REGIONS_CALLBACK_FUNC = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_region_t, # region
    ctypes.py_object) # this is a c_void_p used to wrap a python object

# hsa_status_t (*callback)(hsa_code_object_t code_object, hsa_code_symbol_t symbol, void* data),
HSA_CODE_OBJECT_ITERATE_SYMBOLS_CALLBACK = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_code_object_t,
    hsa_code_symbol_t,
    ctypes.py_object) # this is a c_void_p used to wrap a python object

# hsa_status_t (*alloc_callback)(size_t size, hsa_callback_data_t data, void **address),
HSA_ALLOC_CALLBACK_FUNCTION = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    ctypes.c_size_t,
    hsa_callback_data_t,
    _PTR(ctypes.c_void_p) # this might need to be a ptr to a py_object
    )

void_fn_ptr =  ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p) # this might need to be a ptr to a py_object

# hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data)
HSA_AMD_AGENT_ITERATE_MEMORY_POOLS_CALLBACK = ctypes.CFUNCTYPE(
    hsa_status_t,
    hsa_amd_memory_pool_t,
    ctypes.c_void_p) # this is a c_void_p used to wrap a python object


#------------------------------------------------------------------------------

# Functions used by API calls returning hsa_status_t to check for errors ######

def _build_reverse_error_warn_maps():
    err_map = utils.UniqueDict()
    warn_map = utils.UniqueDict()

    for name in [name for name in dir(enums) if name.startswith('HSA_')]:
        code = getattr(enums, name)
        if 'STATUS_ERROR' in name:
            err_map[code] = name
        elif 'STATUS_INFO' in name:
            warn_map[code] = name
        else:
            pass # should we warn here?
    return err_map, warn_map

ERROR_MAP, WARN_MAP = _build_reverse_error_warn_maps()


def _check_error(result, func, arguments):
    if result != enums.HSA_STATUS_SUCCESS:
        if result >= enums.HSA_STATUS_ERROR:
            errname = ERROR_MAP.get(result, "UNKNOWN_HSA_ERROR")
            msg = "Call to {0} returned {1}".format(func.__name__, errname)
            raise HsaApiError(result, msg)
        else:
            warnname = WARN_MAP.get(result, "UNKNOWN_HSA_INFO")
            msg = "Call to {0} returned {1}".format(func.__name__, warnname)
            warnings.warn(msg, HsaWarning)


# The API prototypes
# These are order based on header files.
API_PROTOTYPES = {

#------------------------------------------------------------------------------
# HSA functions from hsa.h, ordered as per header file.

    # hsa_status_t hsa_status_string(
    #     hsa_status_t status,
    #     const char **status_string);
    'hsa_status_string': {
        'restype': hsa_status_t,
        'argtypes': [hsa_status_t, _PTR(ctypes.c_char_p)],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_init(void)
    'hsa_init': {
        'restype': hsa_status_t,
        'argtypes': [],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_shut_down(void)
    'hsa_shut_down': {
        'restype': hsa_status_t,
        'argtypes': [],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_system_get_info(hsa_system_info_t, void*)
    'hsa_system_get_info': {
        'restype': hsa_status_t,
        'argtypes': [hsa_system_info_t, ctypes.c_void_p],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_system_extension_supported(uint16_t, uint16_t,
    #                                                     uint16_t, bool *);
    'hsa_system_extension_supported': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_uint16,      # extension
                     ctypes.c_uint16,      # version_major
                     ctypes.c_uint16,      # version_minor
                     _PTR(ctypes.c_bool)], # result
        'errcheck': _check_error
    },

    # hsa_status_t hsa_system_get_extension_table(uint16_t, uint16_t,
    #                                             uint16_t, void *);
    'hsa_system_get_extension_table': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_uint16,  # extension
                     ctypes.c_uint16,  # version_major
                     ctypes.c_uint16,  # version_minor
                     ctypes.c_void_p], # result
        'errcheck': _check_error
    },

    # hsa_status_t hsa_agent_get_info(hsa_agent_t, hsa_agent_info_t, void*)
    'hsa_agent_get_info': {
        'restype': hsa_status_t,
        'argtypes': [hsa_agent_t, hsa_agent_info_t, ctypes.c_void_p],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_iterate_agents(hsa_status_t(*)(hsa_agent_t, void*),
    #                                                 void*)
    'hsa_iterate_agents': {
        'restype': hsa_status_t,
        'argtypes': [HSA_ITER_AGENT_CALLBACK_FUNC, ctypes.py_object],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_agent_get_exception_policies(hsa_agent_t agent,
    #                                               hsa_profile_t profile,
    #                                               uint16_t *mask);
    'hsa_agent_get_exception_policies': {
        'restype': hsa_status_t,
        'argtypes': [hsa_agent_t, hsa_profile_t, _PTR(ctypes.c_uint16)],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_agent_extension_supported(uint16_t extension, hsa_agent_t agent,
    #                                           uint16_t version_major,
    #                                           uint16_t version_minor, bool *result);
    'hsa_agent_extension_supported': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_uint16, hsa_agent_t, ctypes.c_uint16, ctypes.c_uint16,
                     _PTR(ctypes.c_bool)],
        'errcheck': _check_error
    },

    #--------------------------------------------------------------------------
    # Signals
    #--------------------------------------------------------------------------

    # hsa_status_t hsa_signal_create(
    #     hsa_signal_value_t initial_value,
    #     uint32_t agent_count,
    #     const hsa_agent_t *agents,
    #     hsa_signal_t *signal)
    'hsa_signal_create': {
        'restype': hsa_status_t,
        'argtypes': [hsa_signal_value_t,
                     ctypes.c_uint32,
                     _PTR(hsa_agent_t),
                     _PTR(hsa_signal_t)],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_signal_destroy(
    #     hsa_signal_t signal)
    'hsa_signal_destroy': {
        'restype': hsa_status_t,
        'argtypes': [hsa_signal_t],
        'errcheck': _check_error
    },

    # hsa_signal_value_t hsa_signal_load_acquire(
    #     hsa_signal_t signal);
    'hsa_signal_load_acquire': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t],
    },

    # hsa_signal_value_t hsa_signal_load_relaxed(
    #     hsa_signal_t signal);
    'hsa_signal_load_relaxed': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t],
    },

    # void hsa_signal_store_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_store_relaxed': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_store_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_store_release': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t],
    },

    # hsa_signal_value_t hsa_signal_exchange_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_acq_rel': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_exchange_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_acquire': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_exchange_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_relaxed': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_exchange_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_release': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_cas_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_acq_rel': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_cas_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_acquire': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_cas_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_relaxed': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_cas_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_release': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
    },

    # void hsa_signal_add_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_acq_rel': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_add_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_acquire': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_add_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_relaxed': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_add_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_release': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_subtract_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_acq_rel': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_subtract_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_acquire': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_subtract_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_relaxed': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_subtract_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_release': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_and_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_acq_rel': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_and_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_acquire': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_and_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_relaxed': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_and_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_release': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_or_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_acq_rel': {
        'restype': None,
        'argtypes': [hsa_signal_t,
                     hsa_signal_value_t]
    },

    # void hsa_signal_or_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_acquire': {
        'restype': None,
        'argtypes': [hsa_signal_t,
                     hsa_signal_value_t]
    },

    # void hsa_signal_or_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_relaxed': {
        'restype': None,
        'argtypes': [hsa_signal_t,
                     hsa_signal_value_t]
    },

    # void hsa_signal_or_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_release': {
        'restype': None,
        'argtypes': [hsa_signal_t,
                     hsa_signal_value_t]
    },

    # void hsa_signal_xor_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_acq_rel': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_xor_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_acquire': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_xor_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_relaxed': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_xor_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_release': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t HSA_API
    #     hsa_signal_wait_acquire(hsa_signal_t signal,
    #                             hsa_signal_condition_t condition,
    #                             hsa_signal_value_t compare_value,
    #                             uint64_t timeout_hint,
    #                             hsa_wait_state_t wait_state_hint);
    'hsa_signal_wait_acquire': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t,
                     hsa_signal_condition_t,
                     hsa_signal_value_t,
                     ctypes.c_uint64,
                     hsa_wait_state_t]
    },

    # hsa_signal_value_t hsa_signal_wait_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_condition_t condition,
    #     hsa_signal_value_t compare_value,
    #     uint64_t timeout_hint,
    #     hsa_wait_state_t wait_state_hint);
    'hsa_signal_wait_relaxed': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t,
                     hsa_signal_condition_t,
                     hsa_signal_value_t,
                     ctypes.c_uint64,
                     hsa_wait_state_t],
    },

    #--------------------------------------------------------------------------
    # Queues
    #--------------------------------------------------------------------------

    # hsa_status_t HSA_API
    # hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type_t type,
    #                  void (*callback)(hsa_status_t status, hsa_queue_t *source,
    #                                   void *data),
    #                  void *data, uint32_t private_segment_size,
    #                  uint32_t group_segment_size, hsa_queue_t **queue);
    'hsa_queue_create': {
        'restype': hsa_status_t,
        'argtypes': [hsa_agent_t,
                     ctypes.c_uint32,
                     hsa_queue_type_t,
                     HSA_QUEUE_CALLBACK_FUNC,
                     ctypes.c_void_p, # data
                     ctypes.c_uint32, # private segment size
                     ctypes.c_uint32, # group segment size
                     _PTR(_PTR(hsa_queue_t))],
        'errcheck': _check_error
    },

    # hsa_status_t
    # hsa_soft_queue_create(hsa_region_t region, uint32_t size,
    #                      hsa_queue_type_t type, uint32_t features,
    #                      hsa_signal_t doorbell_signal, hsa_queue_t **queue);
    'hsa_soft_queue_create': {
        'restype': hsa_status_t,
        'argtypes': [hsa_region_t,
                     ctypes.c_uint32,
                     hsa_queue_type_t,
                     ctypes.c_uint32,
                     hsa_signal_t,
                     _PTR(_PTR(hsa_queue_t))],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_queue_destroy(
    #     hsa_queue_t *queue)
    'hsa_queue_destroy': {
        'restype': hsa_status_t,
        'argtypes': [_PTR(hsa_queue_t)],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_queue_inactivate(hsa_queue_t *queue);
    'hsa_queue_inactivate': {
        'restype': hsa_status_t,
        'argtypes': [_PTR(hsa_queue_t)],
        'errcheck': _check_error
    },

    # uint64_t hsa_queue_load_read_index_acquire(hsa_queue_t *queue);
    'hsa_queue_load_read_index_acquire': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t)]
    },

    # uint64_t hsa_queue_load_read_index_relaxed(hsa_queue_t *queue);
    'hsa_queue_load_read_index_relaxed': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t)]
    },

    # uint64_t hsa_queue_load_write_index_acquire(hsa_queue_t *queue);
    'hsa_queue_load_write_index_acquire': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t)]
    },

    # uint64_t hsa_queue_load_write_index_relaxed(hsa_queue_t *queue);
    'hsa_queue_load_write_index_relaxed': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t)]
    },

    # void hsa_queue_store_write_index_relaxed(hsa_queue_t *queue, uint64_t value);
    'hsa_queue_store_write_index_relaxed': {
        'restype': None,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64]
    },

    # void hsa_queue_store_write_index_release(hsa_queue_t *queue, uint64_t value);
    'hsa_queue_store_write_index_release': {
        'restype': None,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64]
    },

    # uint64_t hsa_queue_cas_write_index_acq_rel(
    #     hsa_queue_t *queue,
    #     uint64_t expected,
    #     uint64_t value);
    'hsa_queue_cas_write_index_acq_rel': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64, ctypes.c_uint64]
    },

    # uint64_t hsa_queue_cas_write_index_acquire(
    #     hsa_queue_t *queue,
    #     uint64_t expected,
    #     uint64_t value);
    'hsa_queue_cas_write_index_acquire': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64, ctypes.c_uint64]
    },

    # uint64_t hsa_queue_cas_write_index_relaxed(
    #     hsa_queue_t *queue,
    #     uint64_t expected,
    #     uint64_t value);
    'hsa_queue_cas_write_index_relaxed': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64, ctypes.c_uint64]
    },

    # uint64_t hsa_queue_cas_write_index_release(
    #     hsa_queue_t *queue,
    #     uint64_t expected,
    #     uint64_t value);
    'hsa_queue_cas_write_index_release': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64, ctypes.c_uint64]
    },

    # uint64_t hsa_queue_add_write_index_acq_rel(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_add_write_index_acq_rel': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64]
    },

    # uint64_t hsa_queue_add_write_index_acquire(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_add_write_index_acquire': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64]
    },

    # uint64_t hsa_queue_add_write_index_relaxed(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_add_write_index_relaxed': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64]
    },

    # uint64_t hsa_queue_add_write_index_release(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_add_write_index_release': {
        'restype': ctypes.c_uint64,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64]
    },

    # void hsa_queue_store_read_index_relaxed(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_store_read_index_relaxed': {
        'restype': None,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64]
    },

    # void hsa_queue_store_read_index_release(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_store_read_index_release': {
        'restype': None,
        'argtypes': [_PTR(hsa_queue_t), ctypes.c_uint64]
    },

    #--------------------------------------------------------------------------
    # Memory
    #--------------------------------------------------------------------------

    # hsa_status_t hsa_region_get_info(
    #     hsa_region_t region,
    #     hsa_region_info_t attribute,
    #     void *value);
    'hsa_region_get_info': {
        'restype': hsa_status_t,
        'argtypes': [hsa_region_t, hsa_region_info_t, ctypes.c_void_p],
        'errcheck': _check_error,
    },

    # hsa_status_t hsa_agent_iterate_regions(
    #     hsa_agent_t agent,
    #     hsa_status_t (*callback)(hsa_region_t region, void *data),
    #     void *data);
    'hsa_agent_iterate_regions': {
        'restype': hsa_status_t,
        'argtypes': [hsa_agent_t,
                     HSA_AGENT_ITERATE_REGIONS_CALLBACK_FUNC,
                     ctypes.py_object],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_memory_allocate(
    #     hsa_region_t region,
    #     size_t size,
    #     void **ptr);
    'hsa_memory_allocate': {
        'restype': hsa_status_t,
        'argtypes': [hsa_region_t, ctypes.c_size_t, _PTR(ctypes.c_void_p)],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_memory_free(
    #     void *ptr);
    'hsa_memory_free': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_void_p],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_memory_copy(
    #     void * dst,
    #     const void * src,
    #     size_t size);
    'hsa_memory_copy': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_memory_assign_agent(void *ptr,
    #                                              hsa_agent_t agent,
    #                                          hsa_access_permission_t access);
    'hsa_memory_assign_agent': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_void_p, hsa_agent_t, hsa_access_permission_t],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_memory_register(
    #     void *address,
    #     size_t size);
    'hsa_memory_register': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_void_p, ctypes.c_size_t],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_memory_deregister(
    #     void *address,
    #     size_t size);
    'hsa_memory_deregister': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_void_p, ctypes.c_size_t],
        'errcheck': _check_error
    },

    #--------------------------------------------------------------------------
    # Code Object functions
    #--------------------------------------------------------------------------

    # hsa_status_t HSA_API hsa_isa_from_name(const char* name,
    #                                        hsa_isa_t* isa);
    'hsa_isa_from_name': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_char_p, _PTR(hsa_isa_t)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_isa_get_info(hsa_isa_t isa,
    #                                       hsa_isa_info_t attribute,
    #                                       uint32_t index,
    #                                       void* value);
    'hsa_isa_get_info': {
        'restype': hsa_status_t,
        'argtypes': [hsa_isa_t, hsa_isa_info_t, ctypes.c_void_p],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_isa_compatible(hsa_isa_t code_object_isa,
    #                                         hsa_isa_t agent_isa,
    #                                         bool* result);
    'hsa_isa_compatible': {
        'restype': hsa_status_t,
        'argtypes': [hsa_isa_t, hsa_isa_t, _PTR(ctypes.c_bool)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_code_object_serialize(
    #    hsa_code_object_t code_object,
    #    hsa_status_t (*alloc_callback)(size_t size,
    #    hsa_callback_data_t data, void **address),
    #    hsa_callback_data_t callback_data,
    #    const char *options,
    #    void **serialized_code_object,
    #    size_t *serialized_code_object_size);
    'hsa_code_object_serialize': {
        'restype': hsa_status_t,
        'argtypes': [HSA_ALLOC_CALLBACK_FUNCTION,
                     hsa_callback_data_t,
                     _PTR(ctypes.c_void_p),
                     hsa_callback_data_t,
                     ctypes.c_char_p,
                     _PTR(ctypes.c_void_p),
                     _PTR(ctypes.c_size_t)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_code_object_deserialize(
    #    void *serialized_code_object,
    #    size_t serialized_code_object_size,
    #    const char *options,
    #    hsa_code_object_t *code_object);
    'hsa_code_object_deserialize': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_void_p,
                     ctypes.c_size_t,
                     ctypes.c_char_p,
                     _PTR(hsa_code_object_t)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_code_object_destroy(
    #    hsa_code_object_t code_object);
    'hsa_code_object_destroy': {
        'restype': hsa_status_t,
        'argtypes': [hsa_code_object_t],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_code_object_get_info(
    #    hsa_code_object_t code_object,
    #    hsa_code_object_info_t attribute,
    #    void *value);
    'hsa_code_object_get_info': {
        'restype': hsa_status_t,
        'argtypes': [hsa_code_object_t,
                     hsa_code_object_info_t,
                     ctypes.c_void_p
                     ],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_code_object_get_symbol(
    #    hsa_code_object_t code_object,
    #    const char *symbol_name,
    #    hsa_code_symbol_t *symbol);
    'hsa_code_object_get_symbol': {
        'restype': hsa_status_t,
        'argtypes': [hsa_code_object_t,
                     ctypes.c_char_p,
                     _PTR(hsa_code_symbol_t)
                     ],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_code_symbol_get_info(
    #    hsa_code_symbol_t code_symbol,
    #    hsa_code_symbol_info_t attribute,
    #    void *value);
    'hsa_code_symbol_get_info': {
        'restype': hsa_status_t,
        'argtypes': [hsa_code_symbol_t,
                     hsa_code_symbol_info_t,
                     ctypes.c_void_p
                     ],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_code_object_iterate_symbols(
    #    hsa_code_object_t code_object,
    #    hsa_status_t (*callback)(hsa_code_object_t code_object, hsa_code_symbol_t symbol, void* data),
    #    void* data);
    'hsa_code_object_iterate_symbols': {
        'restype': hsa_status_t,
        'argtypes': [hsa_code_object_t,
                     HSA_CODE_OBJECT_ITERATE_SYMBOLS_CALLBACK,
                     ctypes.c_void_p
                     ],
        'errcheck': _check_error
    },

    #--------------------------------------------------------------------------
    #  Executable functions
    #--------------------------------------------------------------------------

    # hsa_status_t HSA_API hsa_executable_create(
    #     hsa_profile_t profile,
    #     hsa_executable_state_t executable_state,
    #     const char *options,
    #     hsa_executable_t *executable);

    "hsa_executable_create": {
        'restype': hsa_status_t,
        'argtypes': [hsa_profile_t,
                     hsa_executable_state_t,
                     ctypes.c_char_p,
                     ctypes.POINTER(hsa_executable_t)],
        'errcheck': _check_error,
    },

    # hsa_status_t HSA_API hsa_executable_destroy(
    #     hsa_executable_t executable);

    "hsa_executable_destroy": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_executable_t,
        ],
    },

    # hsa_status_t HSA_API hsa_executable_load_code_object(
    #     hsa_executable_t executable,
    #     hsa_agent_t agent,
    #     hsa_code_object_t code_object,
    #     const char *options);

    "hsa_executable_load_code_object": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_executable_t,
            hsa_agent_t,
            hsa_code_object_t,
            ctypes.c_char_p,
        ],
    },

    # hsa_status_t HSA_API hsa_executable_freeze(
    #     hsa_executable_t executable,
    #     const char *options);

    "hsa_executable_freeze": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_executable_t,
            ctypes.c_char_p,
        ],
    },

    # hsa_status_t HSA_API hsa_executable_get_info(
    #   hsa_executable_t executable,
    #   hsa_executable_info_t attribute,
    #   void *value);
    "hsa_executable_get_info": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_executable_t,
            hsa_executable_info_t,
            ctypes.c_void_p
        ],
    },

    # hsa_status_t HSA_API hsa_executable_global_variable_define(
    #   hsa_executable_t executable,
    #   const char *variable_name,
    #   void *address);
    "hsa_executable_global_variable_define": {
        'restype': hsa_status_t,
        'argtypes': [hsa_executable_t,
                     ctypes.c_char_p,
                     ctypes.c_void_p],
        'errcheck': _check_error,
    },

    # hsa_status_t HSA_API hsa_executable_agent_global_variable_define(
    #   hsa_executable_t executable,
    #   hsa_agent_t agent,
    #   const char *variable_name,
    #   void *address);
    "hsa_executable_agent_global_variable_define": {
        'restype': hsa_status_t,
        'argtypes': [hsa_executable_t,
                     hsa_agent_t,
                     ctypes.c_char_p,
                     ctypes.c_void_p],
        'errcheck': _check_error,
    },

    # hsa_status_t HSA_API hsa_executable_readonly_variable_define(
    #   hsa_executable_t executable,
    #   hsa_agent_t agent,
    #   const char *variable_name,
    #   void *address);
    "hsa_executable_readonly_variable_define": {
        'restype': hsa_status_t,
        'argtypes': [hsa_executable_t,
                     hsa_agent_t,
                     ctypes.c_char_p,
                     ctypes.c_void_p],
        'errcheck': _check_error,
    },

    # hsa_status_t HSA_API hsa_executable_validate(
    #   hsa_executable_t executable,
    #   uint32_t* result);
    "hsa_executable_validate": {
        'restype': hsa_status_t,
        'argtypes': [hsa_executable_t,
                     _PTR(ctypes.c_uint32)],
        'errcheck': _check_error,
    },

    # hsa_status_t HSA_API hsa_executable_get_symbol(
    #     hsa_executable_t executable,
    #     const char *module_name,
    #     const char *symbol_name,
    #     hsa_agent_t agent,
    #     int32_t call_convention,
    #     hsa_executable_symbol_t *symbol);
    "hsa_executable_get_symbol": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_executable_t,
            ctypes.c_char_p,  # module_name (must be NULL for program linkage)
            ctypes.c_char_p,  # symbol_name
            hsa_agent_t,
            ctypes.c_int32,
            ctypes.POINTER(hsa_executable_symbol_t),
        ],
    },

    # hsa_status_t HSA_API hsa_executable_symbol_get_info(
    #     hsa_executable_symbol_t executable_symbol,
    #     hsa_executable_symbol_info_t attribute,
    #     void *value);
    "hsa_executable_symbol_get_info": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_executable_symbol_t,
            hsa_executable_symbol_info_t,
            ctypes.c_void_p,
        ],
    },


    #hsa_status_t HSA_API hsa_executable_iterate_symbols(
    #   hsa_executable_t executable,
    #   hsa_status_t (*callback)(hsa_executable_t executable, hsa_executable_symbol_t symbol, void* data),
    #   void* data);
    "hsa_executable_iterate_symbols": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_executable_symbol_t,
            hsa_executable_symbol_info_t,
            ctypes.c_void_p,
        ],
    },


    #--------------------------------------------------------------------------
    # AMD extensions from hsa_ext_amd.h
    #--------------------------------------------------------------------------

    # hsa_status_t HSA_API hsa_amd_coherency_get_type(hsa_agent_t agent,
    #                                                hsa_amd_coherency_type_t* type);

    "hsa_amd_coherency_get_type": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_agent_t,
            _PTR(hsa_amd_coherency_type_t),
        ],
    },

    # hsa_status_t HSA_API hsa_amd_coherency_set_type(hsa_agent_t agent,
    #                                                hsa_amd_coherency_type_t type);
    "hsa_amd_coherency_get_type": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_agent_t,
            hsa_amd_coherency_type_t,
        ],
    },

    # hsa_status_t HSA_API
    #   hsa_amd_profiling_set_profiler_enabled(hsa_queue_t* queue, int enable);
    "hsa_amd_profiling_set_profiler_enabled": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            _PTR(hsa_queue_t),
            ctypes.c_int,
        ],
    },

    # hsa_status_t HSA_API hsa_amd_profiling_get_dispatch_time(
    #   hsa_agent_t agent, hsa_signal_t signal,
    #   hsa_amd_profiling_dispatch_time_t* time);
    "hsa_amd_profiling_get_dispatch_time": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_agent_t,
            hsa_signal_t,
            _PTR(hsa_amd_profiling_dispatch_time_t)
        ],
    },

    # hsa_status_t HSA_API
    #    hsa_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent,
    #                                                    uint64_t agent_tick,
    #                                                    uint64_t* system_tick);
    "hsa_amd_profiling_convert_tick_to_system_domain": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.c_uint64,
            _PTR(ctypes.c_uint64)
        ],
    },

    # hsa_status_t HSA_API
    # hsa_amd_signal_async_handler(hsa_signal_t signal,
    #                             hsa_signal_condition_t cond,
    #                             hsa_signal_value_t value,
    #                             hsa_amd_signal_handler handler, void* arg);
    "hsa_amd_signal_async_handler": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_signal_t,
            hsa_signal_condition_t,
            hsa_signal_value_t,
            hsa_amd_signal_handler,
            ctypes.c_void_p,
        ],
    },

    #hsa_amd_async_function(void (*callback)(void* arg), void* arg);
    "hsa_amd_async_function": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.POINTER(void_fn_ptr),
            ctypes.c_void_p,
        ],
    },

    #uint32_t HSA_API
    #hsa_amd_signal_wait_any(uint32_t signal_count, hsa_signal_t* signals,
    #                        hsa_signal_condition_t* conds,
    #                        hsa_signal_value_t* values, uint64_t timeout_hint,
    #                        hsa_wait_state_t wait_hint,
    #                        hsa_signal_value_t* satisfying_value);
    "hsa_amd_signal_wait_any": {
        'errcheck': _check_error,
        'restype': ctypes.c_uint32,
        'argtypes': [
            ctypes.c_uint32,
            _PTR(hsa_signal_t),
            _PTR(hsa_signal_condition_t),
            _PTR(hsa_signal_value_t),
            ctypes.c_uint64,
            hsa_wait_state_t,
            _PTR(hsa_signal_value_t),
        ],
    },

    # hsa_status_t HSA_API hsa_amd_image_get_info_max_dim(hsa_agent_t agent,
    #                                               hsa_agent_info_t attribute,
    #                                               void* value);
    "hsa_amd_image_get_info_max_dim": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_agent_t,
            hsa_agent_info_t,
            ctypes.c_void_p,
        ],
    },

    # hsa_status_t HSA_API hsa_amd_queue_cu_set_mask(const hsa_queue_t* queue,
    #                                           uint32_t num_cu_mask_count,
    #                                           const uint32_t* cu_mask);
    "hsa_amd_queue_cu_set_mask": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            _PTR(hsa_queue_t),
            ctypes.c_uint32,
            _PTR(ctypes.c_uint32)
        ],
    },

    # hsa_status_t HSA_API
    # hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
    #                             hsa_amd_memory_pool_info_t attribute,
    #                             void* value);
    "hsa_amd_memory_pool_get_info": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_amd_memory_pool_t,
            hsa_amd_memory_pool_info_t,
            ctypes.c_void_p
        ],
    },

    # hsa_status_t HSA_API hsa_amd_agent_iterate_memory_pools(
    #    hsa_agent_t agent,
    #    hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),
    #    void* data);
    "hsa_amd_agent_iterate_memory_pools": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_agent_t,
            HSA_AMD_AGENT_ITERATE_MEMORY_POOLS_CALLBACK,
            ctypes.c_void_p
        ],
    },

    # hsa_status_t HSA_API hsa_amd_memory_pool_allocate
    #   (hsa_amd_memory_pool_t memory_pool, size_t size,
    #    uint32_t flags, void** ptr);
    "hsa_amd_memory_pool_allocate": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_amd_memory_pool_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            _PTR(ctypes.c_void_p)
        ],
    },

    # hsa_status_t HSA_API hsa_amd_memory_pool_free(void* ptr);
    "hsa_amd_memory_pool_free": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.c_void_p
        ],
    },

    # hsa_status_t HSA_API hsa_amd_memory_async_copy(void* dst,
    #                          hsa_agent_t dst_agent, const void* src,
    #                          hsa_agent_t src_agent, size_t size,
    #                          uint32_t num_dep_signals,
    #                          const hsa_signal_t* dep_signals,
    #                          hsa_signal_t completion_signal);
    "hsa_amd_memory_async_copy": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.c_void_p,
            hsa_agent_t,
            ctypes.c_void_p,
            hsa_agent_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            _PTR(hsa_signal_t),
            hsa_signal_t
        ],
    },

    # hsa_status_t HSA_API hsa_amd_agent_memory_pool_get_info(
    #    hsa_agent_t agent, hsa_amd_memory_pool_t memory_pool,
    #    hsa_amd_agent_memory_pool_info_t attribute, void* value);
    "hsa_amd_agent_memory_pool_get_info": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_agent_t,
            hsa_amd_memory_pool_t,
            hsa_amd_agent_memory_pool_info_t,
            ctypes.c_void_p
        ],
    },


    # hsa_status_t HSA_API
    # hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents,
    #       const uint32_t* flags, const void* ptr);
    "hsa_amd_agents_allow_access": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.c_uint32,
            _PTR(hsa_agent_t),
            _PTR(ctypes.c_uint32),
            ctypes.c_void_p
        ],
    },


    # hsa_status_t HSA_API
    # hsa_amd_memory_pool_can_migrate(hsa_amd_memory_pool_t src_memory_pool,
    #                                hsa_amd_memory_pool_t dst_memory_pool,
    #                                bool* result);
    "hsa_amd_memory_pool_can_migrate": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_amd_memory_pool_t,
            hsa_amd_memory_pool_t,
            _PTR(ctypes.c_bool)
        ],
    },


    # hsa_status_t HSA_API hsa_amd_memory_migrate(const void* ptr,
    #                                            hsa_amd_memory_pool_t memory_pool,
    #                                            uint32_t flags);
    "hsa_amd_memory_migrate": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.c_void_p,
            hsa_amd_memory_pool_t,
            ctypes.c_uint32
        ],
    },


    # hsa_status_t HSA_API hsa_amd_memory_lock(void* host_ptr, size_t size,
    #                                        hsa_agent_t* agents, int num_agent,
    #                                        void** agent_ptr);
    "hsa_amd_memory_lock": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.c_void_p,
            ctypes.c_size_t,
            _PTR(hsa_agent_t),
            ctypes.c_int,
            _PTR(ctypes.c_void_p)
        ],
    },


    # hsa_status_t HSA_API hsa_amd_memory_unlock(void* host_ptr);
    "hsa_amd_memory_unlock": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.c_void_p
        ],
    },


    # hsa_status_t HSA_API
    # hsa_amd_memory_fill(void* ptr, uint32_t value, size_t count);
    "hsa_amd_memory_unlock": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.c_void_p
        ],
    },

    # hsa_status_t HSA_API hsa_amd_interop_map_buffer(uint32_t num_agents,
    #                                        hsa_agent_t* agents,
    #                                        int interop_handle,
    #                                        uint32_t flags,
    #                                        size_t* size,
    #                                        void** ptr,
    #                                        size_t* metadata_size,
    #                                        const void** metadata);
    "hsa_amd_interop_map_buffer": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            ctypes.c_uint32,
            _PTR(hsa_agent_t),
            ctypes.c_int,
            ctypes.c_uint32,
            _PTR(ctypes.c_size_t),
            _PTR(ctypes.c_void_p),
            _PTR(ctypes.c_size_t),
            _PTR(ctypes.c_void_p),
        ],
    },


    # hsa_status_t HSA_API hsa_amd_interop_unmap_buffer(void* ptr);
    "hsa_amd_interop_map_buffer": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            _PTR(ctypes.c_void_p),
        ],
    },


    # hsa_status_t HSA_API hsa_amd_image_create(
    #    hsa_agent_t agent,
    #    const hsa_ext_image_descriptor_t *image_descriptor,
    #    const hsa_amd_image_descriptor_t *image_layout,
    #    const void *image_data,
    #    hsa_access_permission_t access_permission,
    #    hsa_ext_image_t *image
    #    );
    "hsa_amd_image_create": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_agent_t,
            _PTR(hsa_ext_image_descriptor_t),
            _PTR(hsa_amd_image_descriptor_t),
            ctypes.c_void_p,
            hsa_access_permission_t,
            hsa_ext_image_t
        ],
    },

    #--------------------------------------------------------------------------
    # Functions from hsa_ext_finalize.h
    # NOTE: To access these functions use the hsa_ext_finalizer_1_00_pfn_t
    # struct.
    #--------------------------------------------------------------------------

}
