import unittest

import dis
import struct

from numba import jit
from numba.tests.support import TestCase


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
        b[:0] = struct.pack(bytecode_format, dis.EXTENDED_ARG, 1)
        f.__code__ = f.__code__.replace(co_code=bytes(b), co_consts=consts)
        return f

    def test_extended_arg_load_const(self):
        pyfunc = self.get_extended_arg_load_const()
        self.assertPreciseEqual(pyfunc(), 42)
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), 42)


if __name__ == '__main__':
    unittest.main()
