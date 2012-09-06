import unittest
import functools

from numba.decorators import jit as jit_, function

class ByteCodeTestCase(unittest.TestCase):
    jit = staticmethod(jit_)

class ASTTestCase(ByteCodeTestCase):
    jit = staticmethod(lambda *args, **kw: jit_(*args, **dict(kw, backend='ast')))