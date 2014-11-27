from __future__ import print_function, absolute_import, division

import ctypes


hsa_status_t = ctypes.c_int # enum
hsa_system_info_t = ctypes.c_int # enum

hsa_agent_t = ctypes.c_uint64
hsa_agent_feature_t = ctypes.c_int # enum
hsa_device_type_t = ctypes.c_int # enum
hsa_agent_info_t = ctypes.c_int # enum
hsa_system_info_t = ctypes.c_int # enum
hsa_dim3_t = ctypes.c_uint32 * 3 # in fact a x,y,z struct in C
hsa_queue_type_t = ctypes.c_int # enum

HSA_ITER_AGENT_CALLBACK_FUNC = ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t,
                                                ctypes.py_object) 


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
}
