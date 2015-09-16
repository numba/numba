'''
The Device Array API is not implemented in the simulator. This module provides
stubs to allow tests to import correctly.
'''

DeviceRecord = None
from_record_like = None
auto_device = None

def is_cuda_ndarray(obj):
    return getattr(obj, '__cuda_ndarray__', False)
