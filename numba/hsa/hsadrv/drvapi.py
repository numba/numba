from __future__ import print_function, absolute_import, division

import ctypes


hsa_status_t = ctypes.c_int # enum



API_PROTOTYPES = {
    # hsa_status_t hsa_init(void)
    'hsa_init': (hsa_status_t, ),

    # hsa_status_t hsa_shut_down(void)
    'hsa_shut_down': (hsa_status_t, ),
}
