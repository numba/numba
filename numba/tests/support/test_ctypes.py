"""
Test support for ctypes. See also numba.tests.foreign_call.test_ctypes_call.
"""

from numba import *
from numba import environment
from numba.support import ctypes_support

from numba.tests.support import ctypes_values

#-------------------------------------------------------------------
# Utilities
#-------------------------------------------------------------------

env = environment.NumbaEnvironment.get_environment()
context = env.context
from_python = context.typemapper.from_python

def get_cast_type(type):
    assert type.is_cast
    return type.dst_type

def assert_signature(ctypes_func, expected=None):
    sig = from_python(ctypes_func)
    assert sig.is_pointer_to_function
    if expected:
        assert sig.signature == expected, (sig.signature, expected)

#-------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------

rk_state_t = get_cast_type(from_python(ctypes_values.rk_state))

def test_ctypes_func_values():
    signature = int_(rk_state_t.pointer())
    assert_signature(ctypes_values.rk_randomseed, signature)

    # Signature for long_ cannot be reliable with ctypes typedefs (yaay!)
    # signature = void(long_, rk_state_t.pointer())
    assert_signature(ctypes_values.rk_seed) #, signature)

    signature = double(rk_state_t.pointer(), double, double)
    assert_signature(ctypes_values.rk_gamma, signature)

def test_ctypes_data_values():
    assert from_python(ctypes_values.state) == rk_state_t
    assert from_python(ctypes_values.state_p) == rk_state_t.pointer()
    assert from_python(ctypes_values.state_vp) == void.pointer()

if __name__ == '__main__':
    test_ctypes_func_values()
    test_ctypes_data_values()
