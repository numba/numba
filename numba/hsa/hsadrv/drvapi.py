from __future__ import print_function, absolute_import, division

import ctypes
from . import enums

_PTR = ctypes.POINTER

# HSA types ####################################################################

hsa_status_t = ctypes.c_int # enum
hsa_packet_type_t = ctypes.c_int # enum
hsa_queue_type_t = ctypes.c_int # enum
hsa_queue_feature_t = ctypes.c_int # enum
hsa_fence_scope_t = ctypes.c_int # enum
hsa_wait_expectancy_t = ctypes.c_int # enum
hsa_signal_condition_t = ctypes.c_int # enum
hsa_extension_t = ctypes.c_int # enum
hsa_agent_feature_t = ctypes.c_int # enum
hsa_device_type_t = ctypes.c_int # enum
hsa_system_info_t = ctypes.c_int # enum
hsa_agent_info_t = ctypes.c_int # enum
hsa_segment_t = ctypes.c_int # enum
hsa_region_flag_t = ctypes.c_int # enum
hsa_region_info_t = ctypes.c_int # enum

hsa_signal_value_t = ctypes.c_uint64 if enums.HSA_LARGE_MODEL else ctypes.c_uint32

hsa_signal_t = ctypes.c_uint64
hsa_agent_t = ctypes.c_uint64
hsa_region_t = ctypes.c_uint64

# HSA Structures ###############################################################
class hsa_queue_t(ctypes.Structure):
    """In theory, this should be aligned to 64 bytes. In any case, allocation
    of this structure is done by the hsa library"""
    _fields_ = [
        ('type', hsa_queue_type_t),
        ('features', ctypes.c_uint32),
        ('base_address', ctypes.c_uint64),
        ('doorbell_signal', hsa_signal_t),
        ('size', ctypes.c_uint32),
        ('id', ctypes.c_uint32),
        ('service_queue', ctypes.c_uint64)
        ]

class hsa_packet_header_t(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('type', ctypes.c_uint16, 8),
        ('barrier', ctypes.c_uint16, 1),
        ('acquire_fence_scope', ctypes.c_uint16, 2),
        ('release_fence_scope', ctypes.c_uint16, 2),
        ('reserved', ctypes.c_uint16, 3),
    ]

class hsa_dispatch_packet_t(ctypes.Structure):
    """This should be aligned to HSA_PACKET_ALIGN_BYTES (64)"""
    _fields_ = [
        ('header', hsa_packet_header_t),
        ('dimensions', ctypes.c_uint16, 2),
        ('reserved', ctypes.c_uint16, 14),
        ('workgroup_size_x', ctypes.c_uint16),
        ('workgroup_size_y', ctypes.c_uint16),
        ('workgroup_size_z', ctypes.c_uint16),
        ('reserved2', ctypes.c_uint16),
        ('grid_size_x', ctypes.c_uint32),
        ('grid_size_y', ctypes.c_uint32),
        ('grid_size_z', ctypes.c_uint32),
        ('private_segment_size', ctypes.c_uint32),
        ('group_segment_size', ctypes.c_uint32),
        ('kernel_object_address', ctypes.c_uint64),
        ('kerarg_address', ctypes.c_uint64),
        ('reserved3', ctypes.c_uint64),
        ('completion_signal', hsa_signal_t),
    ]

class hsa_agent_dispatch_packet_t(ctypes.Structure):
    """This should be aligned to HSA_PACKET_ALIGN_BYTES (64)"""
    _fields_ = [
        ('header', hsa_packet_header_t),
        ('type', ctypes.c_uint16),
        ('reserved2', ctypes.c_uint32),
        ('return_address', ctypes.c_uint64),
        ('arg', ctypes.c_uint64 * 4),
        ('reserved3', ctypes.c_uint64),
        ('completion_signal', hsa_signal_t),
    ]

class hsa_barrier_packet_t(ctypes.Structure):
    """This should be aligned to HSA_PACKET_ALIGN_BYTES (64)"""
    _fields_ = [
        ('header', hsa_packet_header_t),
        ('reserved2', ctypes.c_uint16),
        ('reserved3', ctypes.c_uint32),
        ('dep_signal', hsa_signal_t * 5),
        ('reserved4', ctypes.c_uint64),
        ('completion_signal', hsa_signal_t),
    ]

# HSA common definitions #######################################################
hsa_powertwo8_t = ctypes.c_uint8
hsa_dim3_t = ctypes.c_uint32 * 3 # in fact a x,y,z struct in C


class hsa_runtime_caller_t(ctypes.Structure):
    _fields_ = [
        ('caller', ctypes.c_uint64),
    ]

hsa_runtime_alloc_data_callback = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_runtime_caller_t, # caller
    ctypes.c_size_t, # byte_size
    _PTR(ctypes.c_void_p)) # address


# finalize types ###############################################################
hsa_ext_brig_profile8_t = ctypes.c_uint8
hsa_ext_brig_machine_model8_t = ctypes.c_uint8
hsa_ext_brig_section_id32_t = ctypes.c_uint32

class hsa_ext_brig_section_header_t(ctypes.Structure):
    _fields_ = [
        ('byte_count', ctypes.c_uint32),
        ('header_byte_count', ctypes.c_uint32),
        ('name_length', ctypes.c_uint32),
        ('name', ctypes.c_char * 1),
    ]

class hsa_ext_brig_module_t(ctypes.Structure):
    # do not directly instantiate this for now.
    # sections is actually a variable lenght array whose lenght is
    # given by section_count (though it is guaranteed to have at
    # least 3). The sections themshelves have structure, but for
    # now we are treating them as opaque.
    #
    # This supports loading Brig-elfs using the elf_utils C module.
    # We will need to improve this in order to support creating our
    # kernels without going to an elf-file.
    _fields_ = [
        ('section_count', ctypes.c_uint32),
        ('sections', ctypes.c_void_p * 3),
    ]

class hsa_ext_brig_module_handle_t(ctypes.Structure):
    _fields_ = [
        ('handle', ctypes.c_uint64),
    ]

hsa_ext_brig_code_section_offset32_t = ctypes.c_uint32
hsa_ext_exception_kind16_t = ctypes.c_uint16
hsa_ext_control_directive_present64_t = ctypes.c_uint64

class hsa_ext_control_directives_t(ctypes.Structure):
    _fields_ = [
        ('enabled_control_directives', hsa_ext_control_directive_present64_t),
        ('enable_break_exceptions', hsa_ext_exception_kind16_t),
        ('enable_detect_exceptions', hsa_ext_exception_kind16_t),
        ('max_dynamic_group_size', ctypes.c_uint32),
        ('max_flat_grid_size', ctypes.c_uint32),
        ('max_flat_workgroup_size', ctypes.c_uint32),
        ('requested_workgroups_per_cu', ctypes.c_uint32),
        ('required_grid_size', hsa_dim3_t),
        ('required_workgroup_size', hsa_dim3_t),
        ('required_dim', ctypes.c_uint8),
        ('reserved', ctypes.c_uint8 * 75),
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
        ('hsail_profile', hsa_ext_brig_profile8_t),
        ('hsail_machine_model', hsa_ext_brig_machine_model8_t),
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
        _PTR(_PTR(hsa_ext_brig_module_t)), # definition_module_brig
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


class hsa_ext_program_handle_t(ctypes.Structure):
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
    hsa_status_t, # return value
    _PTR(hsa_queue_t))

HSA_AGENT_ITERATE_REGIONS_CALLBACK_FUNC = ctypes.CFUNCTYPE(
    hsa_status_t, # return value
    hsa_region_t, # region
    ctypes.py_object) # this is a c_void_p used to wrap a python object

API_PROTOTYPES = {
    # Init/Shutdown ############################################################
    # hsa_status_t hsa_init(void)
    'hsa_init': (
        hsa_status_t, ),

    # hsa_status_t hsa_shut_down(void)
    'hsa_shut_down': (
        hsa_status_t, ),

    # System ###################################################################
    # hsa_status_t hsa_system_get_info(hsa_system_info_t, void*)
    'hsa_system_get_info': (
        hsa_status_t,
        hsa_system_info_t,
        ctypes.c_void_p),

    # Agent ####################################################################
    # hsa_status_t hsa_iterate_agents(hsa_status_t(*)(hsa_agent_t, void*), void*)
    'hsa_iterate_agents': (
        hsa_status_t,
        HSA_ITER_AGENT_CALLBACK_FUNC,
        ctypes.py_object),

    # hsa_status_t hsa_agent_get_info(hsa_agent_t, hsa_agent_info_t, void*)
    'hsa_agent_get_info': (
        hsa_status_t,
        hsa_agent_info_t,
        ctypes.c_void_p),

    # Queues ###################################################################
    # hsa_status_t hsa_queue_create(
    #     hsa_agent_t agent,
    #     uint32_t size,
    #     hsa_queue_type_t tyoe,
    #     void (*callback)(hsa_status_t status, hsa_queue_t *source),
    #     const hsa_queue_t *service_queue,
    #     hsa_queue_t **queue)
    'hsa_queue_create': (
        hsa_status_t,
        hsa_agent_t,
        ctypes.c_uint32,
        hsa_queue_type_t,
        HSA_QUEUE_CALLBACK_FUNC,
        _PTR(hsa_queue_t),
        _PTR(_PTR(hsa_queue_t))),

    # hsa_status_t hsa_queue_destroy(
    #     hsa_queue_t *queue)
    'hsa_queue_destroy': (
        hsa_status_t,
        _PTR(hsa_queue_t)),

    # hsa_status_t hsa_queue_inactivate(hsa_queue_t *queue);
    'hsa_queue_inactivate': (
        hsa_status_t,
        _PTR(hsa_queue_t)),

    # uint64_t hsa_queue_load_read_index_acquire(hsa_queue_t *queue);
    'hsa_queue_load_read_index_acquire': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t)),

    # uint64_t hsa_queue_load_read_index_relaxed(hsa_queue_t *queue);
    'hsa_queue_load_read_index_relaxed': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t)),

    # uint64_t hsa_queue_load_write_index_acquire(hsa_queue_t *queue);
    'hsa_queue_load_write_index_acquire': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t)),

    # uint64_t hsa_queue_load_write_index_relaxed(hsa_queue_t *queue);
    'hsa_queue_load_write_index_relaxed': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t)),

    # void hsa_queue_store_write_index_relaxed(hsa_queue_t *queue, uint64_t value);
    'hsa_queue_store_write_index_relaxed': (
        None,
        _PTR(hsa_queue_t),
        ctypes.c_uint64),

    # void hsa_queue_store_write_index_release(hsa_queue_t *queue, uint64_t value);
    'hsa_queue_store_write_index_release': (
        None,
        _PTR(hsa_queue_t),
        ctypes.c_uint64),

    # uint64_t hsa_queue_cas_write_index_acq_rel(
    #     hsa_queue_t *queue,
    #     uint64_t expected,
    #     uint64_t value);
    'hsa_queue_cas_write_index_acq_rel': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t),
        ctypes.c_uint64,
        ctypes.c_uint64),

    # uint64_t hsa_queue_cas_write_index_acquire(
    #     hsa_queue_t *queue,
    #     uint64_t expected,
    #     uint64_t value);
    'hsa_queue_cas_write_index_acquire': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t),
        ctypes.c_uint64,
        ctypes.c_uint64),

    # uint64_t hsa_queue_cas_write_index_relaxed(
    #     hsa_queue_t *queue,
    #     uint64_t expected,
    #     uint64_t value);
    'hsa_queue_cas_write_index_relaxed': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t),
        ctypes.c_uint64,
        ctypes.c_uint64),

    # uint64_t hsa_queue_cas_write_index_release(
    #     hsa_queue_t *queue,
    #     uint64_t expected,
    #     uint64_t value);
    'hsa_queue_cas_write_index_release': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t),
        ctypes.c_uint64,
        ctypes.c_uint64),

    # uint64_t hsa_queue_add_write_index_acq_rel(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_add_write_index_acq_rel': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t),
        ctypes.c_uint64),

    # uint64_t hsa_queue_add_write_index_acquire(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_add_write_index_acquire': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t),
        ctypes.c_uint64),

    # uint64_t hsa_queue_add_write_index_relaxed(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_add_write_index_relaxed': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t),
        ctypes.c_uint64),

    # uint64_t hsa_queue_add_write_index_release(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_add_write_index_release': (
        ctypes.c_uint64,
        _PTR(hsa_queue_t),
        ctypes.c_uint64),

    # void hsa_queue_store_read_index_relaxed(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_store_read_index_relaxed': (
        None,
        _PTR(hsa_queue_t),
        ctypes.c_uint64),

    # void hsa_queue_store_read_index_release(
    #     hsa_queue_t *queue,
    #     uint64_t value);
    'hsa_queue_store_read_index_release': (
        None,
        _PTR(hsa_queue_t),
        ctypes.c_uint64),

    # Memory ###################################################################

    # hsa_status_t hsa_agent_iterate_regions(
    #     hsa_agent_t agent,
    #     hsa_status_t (*callback)(hsa_region_t region, void *data),
    #     void *data);
    'hsa_agent_iterate_regions': (
        hsa_status_t,
        hsa_agent_t,
        HSA_AGENT_ITERATE_REGIONS_CALLBACK_FUNC,
        ctypes.py_object),

    # hsa_status_t hsa_region_get_info(
    #     hsa_region_t region,
    #     hsa_region_info_t attribute,
    #     void *value);
    'hsa_region_get_info': (
        hsa_status_t,
        hsa_region_t,
        hsa_region_info_t,
        ctypes.c_void_p),

    # hsa_status_t hsa_memory_register(
    #     void *address,
    #     size_t size);
    'hsa_memory_register': (
        hsa_status_t,
        ctypes.c_void_p,
        ctypes.c_size_t),

    # hsa_status_t hsa_memory_deregister(
    #     void *address,
    #     size_t size);
    'hsa_memory_deregister': (
        hsa_status_t,
        ctypes.c_void_p,
        ctypes.c_size_t),

    # hsa_status_t hsa_memory_allocate(
    #     hsa_region_t region,
    #     size_t size,
    #     void **ptr);
    'hsa_memory_allocate': (
        hsa_status_t,
        hsa_region_t,
        ctypes.c_size_t,
        _PTR(ctypes.c_void_p)),

    # hsa_status_t hsa_memory_free(
    #     void *ptr);
    'hsa_memory_free': (
        hsa_status_t,
        ctypes.c_void_p),


    # Signals ##################################################################

    # hsa_status_t hsa_signal_create(
    #     hsa_signal_value_t initial_value,
    #     uint32_t agent_count,
    #     const has_agent_t *agents,
    #     hsa_signal_t *signal)
    'hsa_signal_create': (hsa_status_t, hsa_signal_value_t, ctypes.c_uint32,
                          _PTR(hsa_agent_t), _PTR(hsa_signal_t)),

    # hsa_status_t hsa_signal_destroy(
    #     hsa_signal_t signal)
    'hsa_signal_destroy': (hsa_status_t, hsa_signal_t),

    # hsa_signal_value_t hsa_signal_load_relaxed(
    #     hsa_signal_t signal);
    'hsa_signal_load_relaxed': (
        hsa_signal_value_t,
        hsa_signal_t),

    # hsa_signal_value_t hsa_signal_load_acquire(
    #     hsa_signal_t signal);
    'hsa_signal_load_acquire': (
        hsa_signal_value_t,
        hsa_signal_t),

    # void hsa_signal_store_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_store_relaxed': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_store_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_store_release': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # hsa_signal_value_t hsa_signal_wait_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_condition_t condition,
    #     hsa_signal_value_t compare_value,
    #     uint64_t timeout_hint,
    #     hsa_wait_expectancy_t wait_expectancy_hint);
    'hsa_signal_wait_relaxed': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_condition_t,
        hsa_signal_value_t,
        ctypes.c_uint64,
        hsa_wait_expectancy_t),

    # hsa_signal_value_t hsa_signal_wait_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_condition_t condition,
    #     hsa_signal_value_t compare_value,
    #     uint64_t timeout_hint,
    #     hsa_wait_expectancy_t wait_expectancy_hint);
    'hsa_signal_wait_acquire': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_condition_t,
        hsa_signal_value_t,
        ctypes.c_uint64,
        hsa_wait_expectancy_t),

    # void hsa_signal_and_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_relaxed': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_and_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_acquire': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_and_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_release': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_and_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_and_acq_rel': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_or_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_relaxed': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_or_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_acquire': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_or_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_release': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_or_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_or_acq_rel': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_xor_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_relaxed': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_xor_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_acquire': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_xor_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_release': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_xor_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_xor_acq_rel': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_add_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_relaxed': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_add_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_acquire': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_add_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_release': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_add_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_add_acq_rel': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_subtract_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_relaxed': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_subtract_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_acquire': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_subtract_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_release': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # void hsa_signal_subtract_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_subtract_acq_rel': (
        None,
        hsa_signal_t,
        hsa_signal_value_t),

    # hsa_signal_value_t hsa_signal_exchange_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_relaxed': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_value_t),

    # hsa_signal_value_t hsa_signal_exchange_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_acquire': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_value_t),

    # hsa_signal_value_t hsa_signal_exchange_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_release': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_value_t),

    # hsa_signal_value_t hsa_signal_exchange_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t value);
    'hsa_signal_exchange_acq_rel': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_value_t),

    # hsa_signal_value_t hsa_signal_cas_relaxed(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_relaxed': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_value_t,
        hsa_signal_value_t),

    # hsa_signal_value_t hsa_signal_cas_acquire(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_acquire': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_value_t,
        hsa_signal_value_t),

    # hsa_signal_value_t hsa_signal_cas_release(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_release': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_value_t,
        hsa_signal_value_t),

    # hsa_signal_value_t hsa_signal_cas_acq_rel(
    #     hsa_signal_t signal,
    #     hsa_signal_value_t expected,
    #     hsa_signal_value_t value);
    'hsa_signal_cas_acq_rel': (
        hsa_signal_value_t,
        hsa_signal_t,
        hsa_signal_value_t,
        hsa_signal_value_t),

    # Errors ###################################################################

    # hsa_status_t hsa_status_string(
    #     hsa_status_t status,
    #     const char **status_string);
    'hsa_status_string': (
        hsa_status_t,
        hsa_status_t,
        _PTR(ctypes.c_char_p)),

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
    'hsa_ext_finalize': (
        hsa_status_t,
        hsa_runtime_caller_t,
        hsa_agemt_t,
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
        _PTR(hsa_ext_finalization_handle_t)),

    # hsa_status_t hsa_ext_query_finalization_code_descriptor_count(
    #     hsa_agent_t agent,
    #     hsa_ext_finalization_handle_t finalization,
    #     uint32_t *code_descriptor_count);
    'hsa_ext_query_finalization_code_descriptor_count': (
        hsa_status_t,
        hsa_ext_finalization_handle_t,
        _PTR(ctypes.c_uint32)),

    # hsa_status_t hsa_ext_query_finalization_code_descriptor(
    #     hsa_agent_t agent,
    #     hsa_ext_finalization_handle_t finalization,
    #     uint32_t index,
    #     hsa_ext_code_descriptor_t *code_descriptor);
    'hsa_ext_query_finalization_code_descriptor': (
        hsa_status_t,
        hsa_ext_finalization_handle_t,
        ctypes.c_uint32,
        _PTR(hsa_ext_code_descriptor_t)),

    # hsa_status_t HSA_API hsa_ext_destroy_finalization(
    #     hsa_agent_t agent,
    #     hsa_ext_finalization_handle_t finalization);
    'hsa_ext_destroy_finalization': (
        hsa_status_t,
        hsa_agent_t,
        hsa_ext_finalization_handle_t),

    # hsa_status_t HSA_API hsa_ext_serialize_finalization(
    #     hsa_runtime_caller_t caller,
    #     hsa_agent_t agent,
    #     hsa_ext_finalization_handle_t finalization,
    #     hsa_runtime_alloc_data_callback_t alloc_serialize_data_callback,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     int debug_information,
    #     void *serialized_object);
    'hsa_ext_serialize_finalization': (
        hsa_status_t,
        hsa_runtime_caller_t,
        hsa_agent_t,
        hsa_ext_finalization_handle_t,
        hsa_runtime_alloc_data_callback_t,
        hsa_ext_error_message_callback_t,
        ctypes.c_int,
        ctypes.c_void_p),

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
    'hsa_ext_deserialize_finalization': (
        hsa_status_t,
        hsa_runtime_caller_t,
        ctypes.c_void_p,
        hsa_agent_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        hsa_ext_symbol_address_callback_t,
        hsa_ext_error_message_callback_t,
        ctypes.c_int,
        _PTR(hsa_ext_finalization_handle_t)),

    # linker ###################################################################

    # hsa_status_t hsa_ext_program_create(
    #     hsa_agent_t *agents,
    #     uint32_t agent_count,
    #     hsa_ext_brig_machine_model8_t machine_model,
    #     hsa_ext_brig_profile8_t profile,
    #     hsa_ext_program_handle_t *program);
    'hsa_ext_program_create': (
        hsa_status_t,
        _PTR(hsa_agent_t), ctypes.c_uint32,
        hsa_ext_brig_machine_model8_t,
        hsa_ext_brig_profile8_t,
        _PTR(hsa_ext_program_handle_t)),


    # hsa_status_t HSA_API hsa_ext_program_destroy(
    #     hsa_ext_program_handle_t program);
    'hsa_ext_program_destroy': (
        hsa_status_t,
        hsa_ext_program_handle_t),

    # hsa_status_t hsa_ext_add_module(
    #     hsa_ext_program_handle_t program,
    #     hsa_ext_brig_module_t *brig_module,
    #     hsa_ext_brig_module_handle_t *module);
    'hsa_ext_add_module': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        _PTR(hsa_ext_brig_module_t),
        _PTR(hsa_ext_brig_module_handle_t)),

    # hsa_status_t hsa_ext_finalize_program(
    #     hsa_ext_program_handle_t program,
    #     hsa_agent_t agent,
    #     size_t finalization_request_count,
    #     hsa_ext_finalization_request_t *finalization_request_list,
    #     hsa_ext_control_directives_t *control_directives,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     uint8_t optimization_level,
    #     const char *options,
    #     int debug_information);
    'hsa_ext_finalize_program': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_agent_t,
        ctypes.c_size_t,
        _PTR(hsa_ext_finalization_request_t),
        _PTR(hsa_ext_control_directives_t),
        hsa_ext_error_message_callback_t,
        ctypes.c_uint8,
        ctypes.c_char_p,
        ctypes.c_int),


    # hsa_status_t HSA_API hsa_ext_query_program_agent_count(
    #     hsa_ext_program_handle_t program,
    #     uint32_t *program_agent_count);
    'hsa_ext_query_program_agent_count': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        _PTR(ctypes.c_uint32)),

    # hsa_status_t HSA_API hsa_ext_query_program_agent_id(
    #     hsa_ext_program_handle_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_program_agent_id_t *program_agent_id);
    'hsa_ext_query_program_agent_id': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_agent_t,
        _PTR(hsa_ext_program_agent_id_t)),

    # hsa_status_t HSA_API hsa_ext_query_program_agents(
    #     hsa_ext_program_handle_t program,
    #     uint32_t program_agent_count,
    #     hsa_agent_t *agents);
    'hsa_ext_query_program_agents': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        ctypes.c_uint32,
        _PTR(hsa_agent_t)),

    # hsa_status_t HSA_API hsa_ext_query_program_module_count(
    #     hsa_ext_program_handle_t program,
    #     uint32_t *program_module_count);
    'hsa_ext_query_program_module_count': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        _PTR(ctypes.c_uint32)),

    # hsa_status_t HSA_API hsa_ext_query_program_modules(
    #     hsa_ext_program_handle_t program,
    #     uint32_t program_module_count,
    #     hsa_ext_brig_module_handle_t *modules);
    'hsa_ext_query_program_modules': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        ctypes.c_uint32,
        _PTR(hsa_ext_brig_module_handle_t)),

    # hsa_status_t HSA_API hsa_ext_query_program_brig_module(
    #     hsa_ext_program_handle_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_module_t **brig_module);
    'hsa_ext_query_program_brig_module': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_ext_brig_module_handle_t,
        _PTR(_PTR(hsa_ext_brig_module_t))),

    # hsa_status_t HSA_API hsa_ext_query_call_convention(
    #     hsa_ext_program_handle_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_program_call_convention_id32_t *first_call_convention_id,
    #     uint32_t *call_convention_count);
    'hsa_ext_query_call_convention': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_agent_t,
        _PTR(hsa_ext_program_call_convention_id32_t),
        _PTR(ctypes.c_uint32)),

    # hsa_status_t HSA_API hsa_ext_query_symbol_definition(
    #     hsa_ext_program_handle_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_brig_module_handle_t *definition_module,
    #     hsa_ext_brig_module_t **definition_module_brig,
    #     hsa_ext_brig_code_section_offset32_t *definition_symbol);
    'hsa_ext_query_symbol_definition': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_ext_brig_module_handle_t,
        has_ext_brig_code_section_offset32_t,
        _PTR(hsa_ext_brig_module_handle_t),
        _PTR(_PTR(hsa_ext_brig_module_t)),
        _PTR(hsa_ext_brig_code_section_offset32_t)),

    # hsa_status_t HSA_API hsa_ext_define_program_allocation_global_variable_address(
    #     hsa_ext_program_handle_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     void *address);
    'hsa_ext_define_program_allocation_global_variable_address': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_ext_brig_module_handle_t,
        hsa_ext_brig_code_section_offset32_t,
        hsa_ext_error_message_callback_t,
        ctypes.c_void_p),

    # hsa_status_t HSA_API hsa_ext_query_program_allocation_global_variable_address(
    #     hsa_ext_program_handle_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     void** address);
    'hsa_ext_query_program_allocation_global_variable_address': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_ext_brig_module_handle_t,
        hsa_ext_brig_code_section_offset32_t,
        _PTR(ctypes.c_void_p)),

    # hsa_status_t HSA_API hsa_ext_define_agent_allocation_global_variable_address(
    #     hsa_ext_program_handle_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     void *address);
    'hsa_ext_define_agent_allocation_global_variable_address': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_agent_t,
        hsa_ext_brig_module_handle_t,
        hsa_ext_brig_code_section_offset32_t,
        hsa_ext_error_message_callback_t,
        ctypes.c_void_p),

    # hsa_status_t HSA_API hsa_ext_query_agent_global_variable_address(
    #     hsa_ext_program_handle_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     void** address);
    'hsa_ext_query_agent_global_variable_address': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_agent_t,
        hsa_ext_brig_module_handle_t,
        hsa_brig_code_section_offset32_t,
        _PTR(ctypes.c_void_p)),

    # hsa_status_t HSA_API hsa_ext_define_readonly_variable_address(
    #     hsa_ext_program_handle_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     void* address);
    'hsa_ext_define_readonly_variable_address': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_agent_t,
        hsa_ext_brig_module_handle_t,
        hsa_ext_brig_code_section_offset32_t,
        hsa_ext_error_message_callback_t,
        ctypes.c_void_p),

    # hsa_status_t HSA_API hsa_ext_query_readonly_variable_address(
    #     hsa_ext_program_handle_t program,
    #     hsa_agent_t agent,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     void** address);
    'hsa_ext_query_readonly_variable_address': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_agent_t,
        hsa_ext_brig_module_handle_t,
        hsa_ext_brig_code_section_offset32_t,
        _PTR(ctypes.c_void_p)),

    # hsa_status_t hsa_ext_query_kernel_descriptor_address(
    #     hsa_ext_program_handle_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_code_descriptor_t** kernel_descriptor);
    'hsa_ext_query_kernel_descriptor_address': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_ext_brig_module_handle_t,
        hsa_ext_brig_code_section_offset32_t,
        _PTR(_PTR(hsa_ext_code_descriptor_t))),

    # hsa_status_t HSA_API hsa_ext_query_indirect_function_descriptor_address(
    #     hsa_ext_program_handle_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_brig_code_section_offset32_t symbol,
    #     hsa_ext_code_descriptor_t** indirect_function_descriptor);
    'hsa_ext_query_indirect_function_descriptor_address': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_ext_brig_module_handle_t,
        hsa_ext_brig_code_section_offset32_t,
        _PTR(_PTR(hsa_ext_code_descriptor_t))),

    # hsa_status_t HSA_API hsa_ext_validate_program(
    #     hsa_ext_program_handle_t program,
    #     hsa_ext_error_message_callback_t error_message_callback);
    'hsa_ext_validate_program': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_ext_error_message_callback_t),

    # hsa_status_t HSA_API hsa_ext_validate_program_module(
    #     hsa_ext_program_handle_t program,
    #     hsa_ext_brig_module_handle_t module,
    #     hsa_ext_error_message_callback_t error_message_callback);
    'hsa_ext_validate_program_module': (
        hsa_status_t,
        hsa_ext_program_handle_t,
        hsa_ext_brig_module_handle_t,
        hsa_ext_error_message_callback_t),

    # hsa_status_t HSA_API hsa_ext_serialize_program(
    #     hsa_runtime_caller_t caller,
    #     hsa_ext_program_handle_t program,
    #     hsa_runtime_alloc_data_callback_t alloc_serialize_data_callback,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     int debug_information,
    #     void *serialized_object);
    'hsa_ext_serialize_program': (
        hsa_status_t,
        hsa_runtime_caller_t,
        hsa_ext_program_handle_t,
        hsa_runtime_alloc_data_callback_t,
        hsa_ext_error_message_callback_t,
        ctypes.c_int,
        ctypes.c_void_p),

    # hsa_status_t HSA_API hsa_ext_deserialize_program(
    #     hsa_runtime_caller_t caller,
    #     void *serialized_object,
    #     hsa_ext_program_allocation_symbol_address_t program_allocation_symbol_address,
    #     hsa_ext_agent_allocation_symbol_address_t agent_allocation_symbol_address,
    #     hsa_ext_error_message_callback_t error_message_callback,
    #     int debug_information,
    #     hsa_ext_program_handle_t **program);
    'hsa_ext_deserialize_program': (
        hsa_status_t,
        ctypes.c_void_p,
        hsa_ext_program_allocation_symbol_address_t,
        hsa_ext_agent_allocation_symbol_address_t,
        hsa_ext_error_message_callback_t,
        ctypes.c_int,
        _PTR(_PTR(hsa_ext_program_handle_t))),
}
