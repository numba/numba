import unittest

import dis
import struct
import sys

from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tweak_code


class TestExtendedArg(TestCase):
    """
    Test support for the EXTENDED_ARG opcode.
    """

    def get_extended_arg_load_const(self):
        """
        Get a function with a EXTENDED_ARG opcode before a LOAD_CONST opcode.
        """
        def f():
            x = 5
            return x

        b = bytearray(f.__code__.co_code)
        consts = f.__code__.co_consts
        bytecode_len = 0xff
        bytecode_format = "<BB"
        consts = consts + (None,) * bytecode_len + (42,)
        if utils.PYVERSION >= (3, 11):
            # Python 3.11 has a RESUME op code at the start of a function, need
            # to inject the EXTENDED_ARG after this to influence the LOAD_CONST
            offset = 2 # 2 byte op code
        else:
            offset = 0

        packed_extend_arg = struct.pack(bytecode_format, dis.EXTENDED_ARG, 1)
        b[:] = b[:offset] + packed_extend_arg + b[offset:]
        tweak_code(f, codestring=bytes(b), consts=consts)
        return f

    def test_extended_arg_load_const(self):
        pyfunc = self.get_extended_arg_load_const()
        self.assertPreciseEqual(pyfunc(), 42)
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), 42)



if __name__ == '__main__':
    unittest.main()
