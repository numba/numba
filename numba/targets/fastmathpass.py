from __future__ import absolute_import, print_function

from llvmlite import ir
from llvmlite.ir.transforms import Visitor, CallVisitor


class FastFloatBinOpVisitor(Visitor):
    """
    A pass to add fastmath flag to float-binop instruction if they don't have
    any flags.
    """
    float_binops = frozenset(['fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'fcmp'])

    def visit_Instruction(self, instr):
        if instr.opname in self.float_binops:
            if not instr.flags:
                instr.flags.append('fast')


class FastFloatCallVisitor(CallVisitor):
    """
    A pass to change all float function calls to use fastmath.
    """
    def visit_Call(self, instr):
        # Add to any call that has float/double return type
        if instr.type in (ir.FloatType(), ir.DoubleType()):
            instr.fastmath.add('fast')


def rewrite_module(mod):
    """
    Rewrite the given LLVM module to use fastmath everywhere.
    """
    FastFloatBinOpVisitor().visit(mod)
    FastFloatCallVisitor().visit(mod)

