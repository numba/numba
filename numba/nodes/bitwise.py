import ast
from numba.nodes import *

def is_bitwise(op):
    return isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor))
