import ast
from numba.nodes import *
from numba import typesystem

def is_bitwise(op):
    return isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor,
                           ast.LShift, ast.RShift, ast.Invert))
