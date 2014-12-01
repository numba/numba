from __future__ import print_function, absolute_import, division

import ctypes
from . import enums

hsa_status_t = ctypes.c_int # enum
hsa_system_info_t = ctypes.c_int # enum

hsa_agent_t = ctypes.c_uint64
hsa_signal_t = ctypes.c_uint64
hsa_region_t = ctypes.c_uint64

hsa_agent_feature_t = ctypes.c_int # enum
hsa_device_type_t = ctypes.c_int # enum
hsa_agent_info_t = ctypes.c_int # enum
hsa_system_info_t = ctypes.c_int # enum
hsa_dim3_t = ctypes.c_uint32 * 3 # in fact a x,y,z struct in C
hsa_queue_type_t = ctypes.c_int # enum

hsa_signal_value_t = ctypes.c_uint64 if enums.HSA_LARGE_MODEL else ctypes.c_uint32

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

class hsa_runtime_caller_t(ctypes.Structure):
    _fields_ = [
        ('caller', ctypes.c_uint64),
    ]


HSA_ITER_AGENT_CALLBACK_FUNC = ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t,
                                                ctypes.py_object)

HSA_QUEUE_CALLBACK_FUNC = ctypes.CFUNCTYPE(hsa_status_t,
                                           ctypes.POINTER(hsa_queue_t))


hsa_amd_code_unit_t = ctypes.c_uint64
hsa_ext_symbol_value_callback_t = ctypes.CFUNCTYPE(hsa_status_t,
                                                   hsa_runtime_caller_t,
                                                   ctypes.c_char_p,
                                                   ctypes.POINTER(ctypes.c_uint64))

API_PROTOTYPES = {
    # hsa_status_t hsa_init(void)
    'hsa_init': (hsa_status_t, ),

    # hsa_status_t hsa_shut_down(void)
    'hsa_shut_down': (hsa_status_t, ),

    # hsa_status_t hsa_system_get_info(hsa_system_info_t, void*)
    'hsa_system_get_info': (hsa_status_t, hsa_system_info_t, ctypes.c_void_p),

    # hsa_status_t hsa_iterate_agents(hsa_status_t(*)(hsa_agent_t, void*), void*)
    'hsa_iterate_agents': (hsa_status_t, HSA_ITER_AGENT_CALLBACK_FUNC,
                           ctypes.py_object),

    # hsa_status_t hsa_agent_get_info(hsa_agent_t, hsa_agent_info_t, void*)
    'hsa_agent_get_info': (hsa_status_t, hsa_agent_info_t, ctypes.c_void_p),

    # hsa_status_t hsa_queue_create(
    #     hsa_agent_t agent,
    #     uint32_t size,
    #     hsa_queue_type_t tyoe,
    #     void (*callback)(hsa_status_t status, hsa_queue_t *source),
    #     const hsa_queue_t *service_queue,
    #     hsa_queue_t **queue)
    'hsa_queue_create': (hsa_status_t, hsa_agent_t, ctypes.c_uint32, hsa_queue_type_t,
                         HSA_QUEUE_CALLBACK_FUNC, ctypes.POINTER(hsa_queue_t),
                         ctypes.POINTER(ctypes.POINTER(hsa_queue_t))),

    # hsa_status_t hsa_queue_destroy(
    #     hsa_queue_t *queue)
    'hsa_queue_destroy': (hsa_status_t, ctypes.POINTER(hsa_queue_t)),

    # hsa_status_t hsa_signal_create(
    #     hsa_signal_value_t initial_value,
    #     uint32_t agent_count,
    #     const has_agent_t *agents,
    #     hsa_signal_t *signal)
    'hsa_signal_create': (hsa_status_t, hsa_signal_value_t, ctypes.c_uint32,
                          ctypes.POINTER(hsa_agent_t), ctypes.POINTER(hsa_signal_t)),

    # hsa_status_t hsa_signal_destroy(hsa_signal_t signal)
    'hsa_signal_destroy': (hsa_status_t, hsa_signal_t),



    # AMD extensions

    # code units ###############################################################

    # hsa_status_t hsa_ext_code_unit_load(
    #     hsa_runtime_caller_t caller,
    #     const hsa_agent_t *agent,
    #     size_t agent_count,
    #     void *serialized_code_unit,
    #     const char *options,
    #     hsa_ext_symbol_value_callback_t symbol_value,
    #     hsa_amd_code_unit_t *code_unit)
    'hsa_ext_code_unit_load': (hsa_status_t, hsa_runtime_caller_t,
                               ctypes.POINTER(hsa_agent_t), ctypes.c_size_t,
                               ctypes.c_void_p, ctypes.c_char_p,
                               hsa_ext_symbol_value_callback_t,
                               ctypes.POINTER(hsa_amd_code_unit_t))

    # hsa_status_t hsa_ext_code_unit_destroy(hsa_amd_code_unit_t code_unit)
    'hsa_ext_code_unit_destroy': (hsa_status_t, hsa_amd_code_unit_t)

    # hsa_status_t hsa_ext_code_unit_get_info(
    #     hsa_amd_code_unit_t code_unit,
    #     hsa_amd_code_unit_info_t attribute,
    #     uint32_t index,
    #     void *value)
    'hsa_ext_code_unit_get_info': (hsa_status_t, hsa_amd_code_unit_t,
                                   hsa_amd_code_unit_info_t, ctypes.c_uint32,
                                   ctypes.c_void_p)

}
