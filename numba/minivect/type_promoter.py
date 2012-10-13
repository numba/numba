"""
Promote and demote values of differing types in a minivect AST. This is run
before code generation. In LLVM types need to be equivalent for binary
operations.
"""

import sys
import copy
import functools

import minivisitor
import miniutils
import minitypes
import minierror

comparison_ops = set(['<', '>', '==', '!=', '>=', '<=',])

class TypePromoter(minivisitor.GenericTransform):
    """
    Promote and demote values of differing types.
    """

    def resolve_type(self, node):
        if node.type.is_array:
            node.type = node.type.dtype

    def promote(self, dst_type, node):
        return self.context.astbuilder.promote(dst_type, node)

    def visit_UnopNode(self, node):
        self.resolve_type(node)
        node.operand = self.promote(node.type, self.visit(node.operand))
        return node

    def visit_BinopNode(self, node):
        self.visitchildren(node)
        self.resolve_type(node)

        if node.operator in comparison_ops:
            dst_type = self.context.promote_types(node.lhs.type, node.rhs.type)
        else:
            dst_type = node.type

        if dst_type.is_pointer:
            return node

        return self.handle_binop(dst_type, node)

    def visit_VectorStoreNode(self, node):
        self.visitchildren(node)
        return node

    def handle_binop(self, dst_type, node):
        node.lhs = self.promote(dst_type, node.lhs)
        node.rhs = self.promote(dst_type, node.rhs)
        return node

    def visit_AssignmentExpr(self, node):
        self.visitchildren(node)
        self.resolve_type(node)
        return self.handle_binop(node.lhs.type, node)

    def visit_ResolvedVariable(self, node):
        return self.visit(node.element)