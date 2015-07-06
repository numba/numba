from __future__ import print_function, absolute_import, division

import ctypes
import warnings

from ... import utils

from . import enums
from .error import HsaApiError, HsaWarning

_PTR = ctypes.POINTER

# HSA types ####################################################################

hsa_status_t = ctypes.c_int # enum
hsa_packet_type_t = ctypes.c_int # enum
hsa_queue_type_t = ctypes.c_int # enum
hsa_queue_feature_t = ctypes.c_int # enum
hsa_fence_scope_t = ctypes.c_int # enum
hsa_wait_state_t = ctypes.c_int # enum
hsa_signal_condition_t = ctypes.c_int # enum
hsa_extension_t = ctypes.c_int # enum
hsa_agent_feature_t = ctypes.c_int # enum
hsa_device_type_t = ctypes.c_int # enum
hsa_system_info_t = ctypes.c_int # enum
hsa_agent_info_t = ctypes.c_int # enum
hsa_region_segment_t = ctypes.c_int # enum
hsa_region_flag_t = ctypes.c_int # enum
hsa_region_info_t = ctypes.c_int # enum
hsa_executable_state_t = ctypes.c_int # enum
hsa_executable_symbol_info_t = ctypes.c_int # enum

hsa_signal_value_t = ctypes.c_uint64 if enums.HSA_LARGE_MODEL else ctypes.c_uint32

hsa_signal_t = ctypes.c_uint64
hsa_agent_t = ctypes.c_uint64
hsa_region_t = ctypes.c_uint64

hsa_uintptr_t = ctypes.c_uint64

hsa_isa_t = ctypes.c_uint64

hsa_code_object_type_t = ctypes.c_int
hsa_code_object_t = ctypes.c_uint64

hsa_executable_t = ctypes.c_uint64
hsa_executable_symbol_t = ctypes.c_uint64

# HSA Structures ###############################################################
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
        # ifdef HSA_LARGE_MODEL
        ('kernarg_address', ctypes.c_uint64),
        # SMALL Machine has a reversed uint32
        ('reserved2', ctypes.c_uint64), # Must be zero
        ('completion_signal', hsa_signal_t),
    ]
#
# class hsa_agent_dispatch_packet_t(ctypes.Structure):
#     """This should be aligned to HSA_PACKET_ALIGN_BYTES (64)"""
#     _fields_ = [
#         ('header', hsa_packet_header_t),
#         ('type', ctypes.c_uint16),
#         ('reserved2', ctypes.c_uint32),
#         ('return_address', ctypes.c_uint64),
#         ('arg', ctypes.c_uint64 * 4),
#         ('reserved3', ctypes.c_uint64),
#         ('completion_signal', hsa_signal_t),
#     ]
#
# class hsa_barrier_packet_t(ctypes.Structure):
#     """This should be aligned to HSA_PACKET_ALIGN_BYTES (64)"""
#     _fields_ = [
#         ('header', hsa_packet_header_t),
#         ('reserved2', ctypes.c_uint16),
#         ('reserved3', ctypes.c_uint32),
#         ('dep_signal', hsa_signal_t * 5),
#         ('reserved4', ctypes.c_uint64),
#         ('completion_signal', hsa_signal_t),
#     ]

# HSA common definitions #######################################################
hsa_powertwo8_t = ctypes.c_uint8
hsa_dim3_t = ctypes.c_uint32 * 3 # in fact a x,y,z struct in C


class hsa_runtime_caller_t(ctypes.Structure):
    _fields_ = [
        ('caller', ctypes.c_uint64),
    ]

hsa_runtime_alloc_data_callback_t = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_runtime_caller_t, # caller
    ctypes.c_size_t, # byte_size
    _PTR(ctypes.c_void_p)) # address


# finalize types ###############################################################
hsa_profile_t = ctypes.c_uint32
hsa_machine_model_t = ctypes.c_uint32
hsa_ext_brig_section_id32_t = ctypes.c_uint32
hsa_default_float_rounding_mode_t = ctypes.c_uint32

class hsa_ext_brig_section_header_t(ctypes.Structure):
    _fields_ = [
        ('byte_count', ctypes.c_uint32),
        ('header_byte_count', ctypes.c_uint32),
        ('name_length', ctypes.c_uint32),
        ('name', ctypes.c_char * 1),
    ]

hsa_ext_module_t = ctypes.c_void_p

class hsa_ext_brig_module_handle_t(ctypes.Structure):
    _fields_ = [
        ('handle', ctypes.c_uint64),
    ]

hsa_ext_brig_code_section_offset32_t = ctypes.c_uint32
hsa_ext_exception_kind16_t = ctypes.c_uint16
hsa_ext_control_directive_present64_t = ctypes.c_uint64

class hsa_ext_control_directives_t(ctypes.Structure):
    _fields_ = [
        ('control_directives_mask', hsa_ext_control_directive_present64_t),
        ('break_exceptions_mask', hsa_ext_exception_kind16_t),
        ('detect_exceptions_mask', hsa_ext_exception_kind16_t),
        ('max_dynamic_group_size', ctypes.c_uint32),
        ('max_flat_grid_size', ctypes.c_uint64),
        ('max_flat_workgroup_size', ctypes.c_uint32),
        ('reserved1', ctypes.c_uint32),
        ('required_grid_size', ctypes.c_uint64),
        ('required_workgroup_size', hsa_dim3_t),
        ('required_dim', ctypes.c_uint8),
        ('reserved2', ctypes.c_uint8 * 75),
    ]

hsa_ext_code_kind32_t = ctypes.c_uint32
hsa_ext_program_call_convention_id32_t = ctypes.c_uint32

class hsa_ext_code_handle_t(ctypes.Structure):
    _fields_ = [
        ('handle', ctypes.c_uint64),
    ]

class hsa_ext_debug_information_handle_t(ctypes.Structure):
    _fields_ = [
        ('handle', ctypes.c_uint64),
    ]

class hsa_ext_code_descriptor_t(ctypes.Structure):
    _fields_ = [
        ('code_type', hsa_ext_code_kind32_t),
        ('workgroup_group_segment_byte_size', ctypes.c_uint32),
        ('kernarg_segment_byte_size', ctypes.c_uint64),
        ('workitem_private_segment_byte_size', ctypes.c_uint32),
        ('workgroup_fbarrier_count', ctypes.c_uint32),
        ('code', hsa_ext_code_handle_t),
        ('kernarg_segment_alignment', hsa_powertwo8_t),
        ('group_segment_alignment', hsa_powertwo8_t),
        ('private_segment_alignment', hsa_powertwo8_t),
        ('wavefront_size', hsa_powertwo8_t),
        ('program_call_convention', hsa_ext_program_call_convention_id32_t),
        ('module', hsa_ext_brig_module_handle_t),
        ('symbol', hsa_ext_brig_code_section_offset32_t),
        ('hsail_profile', hsa_profile_t),
        ('hsail_machine_model', hsa_machine_model_t),
        ('_reserved1', ctypes.c_uint16),
        ('debug_information', hsa_ext_debug_information_handle_t),
        ('agent_vendor', ctypes.c_char * 24),
        ('agent_name', ctypes.c_char * 24),
        ('hsail_version_major', ctypes.c_uint32),
        ('hsail_version_minor', ctypes.c_uint32),
        ('_reserved2', ctypes.c_uint64),
        ('control_directive', hsa_ext_control_directives_t),
    ]

class hsa_ext_finalization_request_t(ctypes.Structure):
    _fields_ = [
        ('module', hsa_ext_brig_module_handle_t),
        ('symbol', hsa_ext_brig_code_section_offset32_t),
        ('program_call_convention', hsa_ext_program_call_convention_id32_t),
    ]

class hsa_ext_finalization_handle_t(ctypes.Structure):
    _fields_ = [
        ('handle', ctypes.c_uint64),
    ]

hsa_ext_symbol_definition_callback_t = ctypes.CFUNCTYPE(
        hsa_status_t,  # return value
        hsa_runtime_caller_t,  # caller
        hsa_ext_brig_module_handle_t, # module
        hsa_ext_brig_code_section_offset32_t, # symbol
        _PTR(hsa_ext_brig_module_handle_t), # definition module
        _PTR(_PTR(hsa_ext_module_t)), # definition_module_brig
        _PTR(hsa_ext_brig_code_section_offset32_t)
)

hsa_ext_symbol_address_callback_t = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_runtime_caller_t, # caller
    hsa_ext_brig_module_handle_t, # module
    hsa_ext_brig_code_section_offset32_t, # symbol
    _PTR(ctypes.c_uint64) # symbol address
)

hsa_ext_error_message_callback_t = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_runtime_caller_t, # caller
    hsa_ext_brig_module_handle_t, # module
    hsa_ext_brig_code_section_offset32_t, # statement
    ctypes.c_int32, # indent_level
    ctypes.c_char_p # message
)

hsa_ext_program_allocation_symbol_address_t = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_runtime_caller_t, # caller
    ctypes.c_char_p, # name
    _PTR(ctypes.c_uint64)) # symbol_address

hsa_ext_agent_allocation_symbol_address_t = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_runtime_caller_t, # caller
    hsa_agent_t, # agent
    ctypes.c_char_p, # name
    _PTR(ctypes.c_uint64)) # symbol address



class hsa_ext_brig_module_handle_t(ctypes.Structure):
    _fields_ = [
        ('handle', ctypes.c_uint64),
    ]


class hsa_ext_program_t(ctypes.Structure):
    _fields_ = [
        ('handle', ctypes.c_uint64),
    ]

hsa_ext_program_agent_id_t = ctypes.c_uint32


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


# Function used by API calls returning hsa_status_t to check for errors ########

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
API_PROTOTYPES = {
    # Init/Shutdown ############################################################
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

    # System ###################################################################
    # hsa_status_t hsa_system_get_info(hsa_system_info_t, void*)
    'hsa_system_get_info': {
        'restype': hsa_status_t,
        'argtypes': [hsa_system_info_t, ctypes.c_void_p],
        'errcheck': _check_error
    },

    # Agent ####################################################################
    # hsa_status_t hsa_iterate_agents(hsa_status_t(*)(hsa_agent_t, void*), void*)
    'hsa_iterate_agents': {
        'restype': hsa_status_t,
        'argtypes': [HSA_ITER_AGENT_CALLBACK_FUNC, ctypes.py_object],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_agent_get_info(hsa_agent_t, hsa_agent_info_t, void*)
    'hsa_agent_get_info': {
        'restype': hsa_status_t,
        'argtypes': [hsa_agent_info_t, ctypes.c_void_p],
        'errcheck': _check_error
    },

    # Queues ###################################################################
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

    # Memory ###################################################################

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

    # hsa_status_t hsa_region_get_info(
    #     hsa_region_t region,
    #     hsa_region_info_t attribute,
    #     void *value);
    'hsa_region_get_info': {
        'restype': hsa_status_t,
        'argtypes': [hsa_region_t, hsa_region_info_t, ctypes.c_void_p],
        'errcheck': _check_error,
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


    # Signals ##################################################################

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


    # hsa_signal_value_t hsa_signal_load_relaxed(
    #     hsa_signal_t signal);
    'hsa_signal_load_relaxed': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t],
    },

    # hsa_signal_value_t hsa_signal_load_acquire(
    #     hsa_signal_t signal);
    'hsa_signal_load_acquire': {
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

    # # hsa_signal_value_t hsa_signal_wait_relaxed(
    # #     hsa_signal_t signal,
    # #     hsa_signal_condition_t condition,
    # #     hsa_signal_value_t compare_value,
    # #     uint64_t timeout_hint,
    # #     hsa_wait_expectancy_t wait_expectancy_hint);
    # 'hsa_signal_wait_relaxed': {
    #     'restype': hsa_signal_value_t,
    #     'argtypes': [hsa_signal_t,
    #                  hsa_signal_condition_t,
    #                  hsa_signal_value_t,
    #                  ctypes.c_uint64,
    #                  hsa_wait_expectancy_t],
    # },

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

    # void hsa_signal_and_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_relaxed': {
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

    # void hsa_signal_and_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_release': {
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

    # void hsa_signal_or_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_relaxed': {
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

    # void hsa_signal_or_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_release': {
        'restype': None,
        'argtypes': [hsa_signal_t,
                     hsa_signal_value_t]
    },

    # void hsa_signal_or_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_acq_rel': {
        'restype': None,
        'argtypes': [hsa_signal_t,
                     hsa_signal_value_t]
    },

    # void hsa_signal_xor_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_relaxed': {
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

    # void hsa_signal_xor_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_release': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # void hsa_signal_xor_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_acq_rel': {
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

    # void hsa_signal_add_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_acquire': {
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

    # void hsa_signal_add_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_acq_rel': {
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

    # void hsa_signal_subtract_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_acquire': {
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

    # void hsa_signal_subtract_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_acq_rel': {
        'restype': None,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_exchange_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_relaxed': {
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

    # hsa_signal_value_t hsa_signal_exchange_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_release': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_exchange_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_acq_rel': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_cas_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_relaxed': {
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

    # hsa_signal_value_t hsa_signal_cas_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_release': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
    },

    # hsa_signal_value_t hsa_signal_cas_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_acq_rel': {
        'restype': hsa_signal_value_t,
        'argtypes': [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
    },

    # Errors ###################################################################

    # hsa_status_t hsa_status_string(
    #     hsa_status_t status,
    #     const char **status_string);
    'hsa_status_string': {
        'restype': hsa_status_t,
        'argtypes': [hsa_status_t, _PTR(ctypes.c_char_p)],
        'errcheck': _check_error
    },

    # Extensions ###############################################################
    # Not yet implemented in the underlying library

    # AMD extensions

    # finalization #############################################################

    # hsa_status_t HSA_API hsa_ext_finalize(
    #     hsa_runtime_caller_t caller,
    #     hsa_agent_t agent,
    #     uint32_t program_agent_id,
    #     uint32_t program_agent_count,
    #     size_t finalization_request_count,
    #     hsa_ext_finalization_request_t *finalization_request_list,
    #     hsa_ext_control_directives_t *control_directives,
    #     hsa_ext_symbol_definition_callback_t symbol_definition_callback,
    #     hsa_ext_symbol_address_callback_t symbol_address_callback,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     uint8_t optimization_level,
    #     const char *options,
    #     int debug_information,
    #     hsa_ext_finalization_handle_t *finalization);
    'hsa_ext_finalize': {
        'restype': hsa_status_t,
        'argtypes': [hsa_runtime_caller_t,
                     hsa_agent_t,
                     ctypes.c_uint32,
                     ctypes.c_uint32,
                     ctypes.c_size_t,
                     _PTR(hsa_ext_finalization_request_t),
                     _PTR(hsa_ext_control_directives_t),
                     hsa_ext_symbol_definition_callback_t,
                     hsa_ext_symbol_address_callback_t,
                     hsa_ext_error_message_callback_t,
                     ctypes.c_uint8,
                     ctypes.c_char_p,
                     ctypes.c_int,
                     _PTR(hsa_ext_finalization_handle_t)],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_ext_query_finalization_code_descriptor_count(
    #     hsa_agent_t agent,
    #     hsa_ext_finalization_handle_t finalization,
    #     uint32_t *code_descriptor_count);
    'hsa_ext_query_finalization_code_descriptor_count': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_finalization_handle_t, _PTR(ctypes.c_uint32)],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_ext_query_finalization_code_descriptor(
    #     hsa_agent_t agent,
    #     hsa_ext_finalization_handle_t finalization,
    #     uint32_t index,
    #     hsa_ext_code_descriptor_t *code_descriptor);
    'hsa_ext_query_finalization_code_descriptor': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_finalization_handle_t,
                     ctypes.c_uint32,
                     _PTR(hsa_ext_code_descriptor_t)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_destroy_finalization(
    #     hsa_agent_t agent,
    #     hsa_ext_finalization_handle_t finalization);
    'hsa_ext_destroy_finalization': {
        'restype': hsa_status_t,
        'argtypes': [hsa_agent_t, hsa_ext_finalization_handle_t],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_serialize_finalization(
    #     hsa_runtime_caller_t caller,
    #     hsa_agent_t agent,
    #     hsa_ext_finalization_handle_t finalization,
    #     hsa_runtime_alloc_data_callback_t alloc_serialize_data_callback,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     int debug_information,
    #     void *serialized_object);
    'hsa_ext_serialize_finalization': {
        'restype': hsa_status_t,
        'argtypes': [hsa_runtime_caller_t,
                     hsa_agent_t,
                     hsa_ext_finalization_handle_t,
                     hsa_runtime_alloc_data_callback_t,
                     hsa_ext_error_message_callback_t,
                     ctypes.c_int,
                     ctypes.c_void_p],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_deserialize_finalization(
    #     hsa_runtime_caller_t caller,
    #     void *serialized_object,
    #     hsa_agent_t agent,
    #     uint32_t program_agent_id,
    #     uint32_t program_agent_count,
    #     hsa_ext_symbol_address_callback_t symbol_address_callback,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     int debug_information,
    #     hsa_ext_finalization_handle_t *finalization);
    'hsa_ext_deserialize_finalization': {
        'restype': hsa_status_t,
        'argtypes': [hsa_runtime_caller_t,
                     ctypes.c_void_p,
                     hsa_agent_t,
                     ctypes.c_uint32,
                     ctypes.c_uint32,
                     hsa_ext_symbol_address_callback_t,
                     hsa_ext_error_message_callback_t,
                     ctypes.c_int,
                     _PTR(hsa_ext_finalization_handle_t)],
        'errcheck': _check_error
    },

    # linker ###################################################################

    # hsa_status_t HSA_API hsa_ext_program_create(
    #             hsa_machine_model_t machine_model,
    #             hsa_profile_t profile,
    #             hsa_default_float_rounding_mode_t default_float_rounding_mode,
    #             const char *options,
    #             hsa_ext_program_t *program);
    'hsa_ext_program_create': {
        'restype': hsa_status_t,
        'argtypes': [hsa_machine_model_t,
                     hsa_profile_t,
                     hsa_default_float_rounding_mode_t,
                     ctypes.c_char_p,
                     _PTR(hsa_ext_program_t)],
        'errcheck': _check_error
    },


    # hsa_status_t HSA_API hsa_ext_program_destroy(
    #     hsa_ext_program_t program);
    'hsa_ext_program_destroy': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_program_add_module(
    #     hsa_ext_program_t program,
    #     hsa_ext_module_t module);
    'hsa_ext_program_add_module': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_ext_module_t],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_program_finalize(
    #     hsa_ext_program_t program,
    #     hsa_isa_t isa,
    #     int32_t call_convention,
    #     hsa_ext_control_directives_t control_directives,
    #     const char *options,
    #     hsa_code_object_type_t code_object_type,
    #     hsa_code_object_t *code_object);

    'hsa_ext_program_finalize': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_isa_t,
                     ctypes.c_int32,
                     hsa_ext_control_directives_t,
                     ctypes.c_char_p,
                     hsa_code_object_type_t,
                     _PTR(hsa_code_object_t),
                     ],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_program_agent_count(
    #     hsa_ext_program_t program,
    #     uint32_t *program_agent_count);
    'hsa_ext_query_program_agent_count': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     _PTR(ctypes.c_uint32)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_program_agent_id(
    #     hsa_ext_program_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_program_agent_id_t *program_agent_id);
    'hsa_ext_query_program_agent_id': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_agent_t,
                     _PTR(hsa_ext_program_agent_id_t)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_program_agents(
    #     hsa_ext_program_t program,
    #     uint32_t program_agent_count,
    #     hsa_agent_t *agents);
    'hsa_ext_query_program_agents': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     ctypes.c_uint32,
                     _PTR(hsa_agent_t)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_program_module_count(
    #     hsa_ext_program_t program,
    #     uint32_t *program_module_count);
    'hsa_ext_query_program_module_count': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t, _PTR(ctypes.c_uint32)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_program_modules(
    #     hsa_ext_program_t program,
    #     uint32_t program_module_count,
    #     hsa_ext_brig_module_handle_t *modules);
    'hsa_ext_query_program_modules': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     ctypes.c_uint32,
                     _PTR(hsa_ext_brig_module_handle_t)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_program_brig_module(
    #     hsa_ext_program_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_module_t **brig_module);
    'hsa_ext_query_program_brig_module': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_ext_brig_module_handle_t,
                     _PTR(_PTR(hsa_ext_module_t))],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_call_convention(
    #     hsa_ext_program_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_program_call_convention_id32_t *first_call_convention_id,
    #     uint32_t *call_convention_count);
    'hsa_ext_query_call_convention': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_agent_t,
                     _PTR(hsa_ext_program_call_convention_id32_t),
                     _PTR(ctypes.c_uint32)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_symbol_definition(
    #     hsa_ext_program_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_brig_module_handle_t *definition_module,
    #     hsa_ext_module_t **definition_module_brig,
    #     hsa_ext_brig_code_section_offset32_t *definition_symbol);
    'hsa_ext_query_symbol_definition': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_brig_code_section_offset32_t,
                     _PTR(hsa_ext_brig_module_handle_t),
                     _PTR(_PTR(hsa_ext_module_t)),
                     _PTR(hsa_ext_brig_code_section_offset32_t)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_define_program_allocation_global_variable_address(
    #     hsa_ext_program_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     void *address);
    'hsa_ext_define_program_allocation_global_variable_address': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_brig_code_section_offset32_t,
                     hsa_ext_error_message_callback_t,
                     ctypes.c_void_p],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_program_allocation_global_variable_address(
    #     hsa_ext_program_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     void** address);
    'hsa_ext_query_program_allocation_global_variable_address': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_brig_code_section_offset32_t,
                     _PTR(ctypes.c_void_p)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_define_agent_allocation_global_variable_address(
    #     hsa_ext_program_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     void *address);
    'hsa_ext_define_agent_allocation_global_variable_address': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_agent_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_brig_code_section_offset32_t,
                     hsa_ext_error_message_callback_t,
                     ctypes.c_void_p],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_agent_global_variable_address(
    #     hsa_ext_program_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     void** address);
    'hsa_ext_query_agent_global_variable_address': {
        'retype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_agent_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_brig_code_section_offset32_t,
                     _PTR(ctypes.c_void_p)],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_define_readonly_variable_address(
    #     hsa_ext_program_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     void* address);
    'hsa_ext_define_readonly_variable_address': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_agent_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_brig_code_section_offset32_t,
                     hsa_ext_error_message_callback_t,
                     ctypes.c_void_p],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_readonly_variable_address(
    #     hsa_ext_program_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     void** address);
    'hsa_ext_query_readonly_variable_address': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_agent_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_brig_code_section_offset32_t,
                     _PTR(ctypes.c_void_p)],
        'errcheck': _check_error
    },

    # hsa_status_t hsa_ext_query_kernel_descriptor_address(
    #     hsa_ext_program_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_code_descriptor_t** kernel_descriptor);
    'hsa_ext_query_kernel_descriptor_address': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_brig_code_section_offset32_t,
                     _PTR(_PTR(hsa_ext_code_descriptor_t))],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_query_indirect_function_descriptor_address(
    #     hsa_ext_program_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_code_descriptor_t** indirect_function_descriptor);
    'hsa_ext_query_indirect_function_descriptor_address': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_brig_code_section_offset32_t,
                     _PTR(_PTR(hsa_ext_code_descriptor_t))],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_validate_program(
    #     hsa_ext_program_t program,
    #     hsa_ext_error_message_callback_t error_message_callback);
    'hsa_ext_validate_program': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t, hsa_ext_error_message_callback_t],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_validate_program_module(
    #     hsa_ext_program_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_error_message_callback_t error_message_callback);
    'hsa_ext_validate_program_module': {
        'restype': hsa_status_t,
        'argtypes': [hsa_ext_program_t,
                     hsa_ext_brig_module_handle_t,
                     hsa_ext_error_message_callback_t],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_serialize_program(
    #     hsa_runtime_caller_t caller,
    #     hsa_ext_program_t program,
    #     hsa_runtime_alloc_data_callback_t alloc_serialize_data_callback,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     int debug_information,
    #     void *serialized_object);
    'hsa_ext_serialize_program': {
        'restype': hsa_status_t,
        'argtypes': [hsa_runtime_caller_t,
                     hsa_ext_program_t,
                     hsa_runtime_alloc_data_callback_t,
                     hsa_ext_error_message_callback_t,
                     ctypes.c_int,
                     ctypes.c_void_p],
        'errcheck': _check_error
    },

    # hsa_status_t HSA_API hsa_ext_deserialize_program(
    #     hsa_runtime_caller_t caller,
    #     void *serialized_object,
    #     hsa_ext_program_allocation_symbol_address_t program_allocation_symbol_address,
    #     hsa_ext_agent_allocation_symbol_address_t agent_allocation_symbol_address,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     int debug_information,
    #     hsa_ext_program_t **program);
    'hsa_ext_deserialize_program': {
        'restype': hsa_status_t,
        'argtypes': [ctypes.c_void_p,
                     hsa_ext_program_allocation_symbol_address_t,
                     hsa_ext_agent_allocation_symbol_address_t,
                     hsa_ext_error_message_callback_t,
                     ctypes.c_int,
                     _PTR(_PTR(hsa_ext_program_t))],
        'errcheck': _check_error
    },

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

    # hsa_status_t HSA_API hsa_executable_destroy(
    #     hsa_executable_t executable);

    "hsa_executable_destroy": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_executable_t,
        ],
    },

    # hsa_status_t HSA_API hsa_code_object_destroy(
    #     hsa_code_object_t code_object);

    "hsa_code_object_destroy": {
        'errcheck': _check_error,
        'restype': hsa_status_t,
        'argtypes': [
            hsa_code_object_t,
        ],
    },
}
