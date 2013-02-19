import ast
from functools import reduce

import numba
from numba import *
from numba import error
from numba import visitors, nodes
from numba import function_util
from numba.symtab import Variable
from numba import pyconsts

logger = logging.getLogger(__name__)

opmap = {
    ast.Eq        : pyconsts.Py_EQ,
    ast.NotEq     : pyconsts.Py_NE,
    ast.Lt        : pyconsts.Py_LT,
    ast.LtE       : pyconsts.Py_LE,
    ast.Gt        : pyconsts.Py_GT,
    ast.GtE       : pyconsts.Py_GE,
}

def build_boolop(right, left):
    node = ast.BoolOp(ast.And(), [left, right])
    return nodes.typednode(node, bool_)

class SpecializeComparisons(visitors.NumbaTransformer):
    """
    Rewrite cascaded ast.Compare nodes to a sequence of boolean operations
    ANDed together:

        a < b < c

    becomes

        a < b and b < c
    """

    def single_compare(self, node):
        if node.left.type.is_object:
            return self.single_compare_objects(node)

        if node.left.type.is_pointer and node.comparators[0].type.is_pointer:
            # Coerce pointers to integer values before comparing
            node.left = nodes.CoercionNode(node.left, Py_uintptr_t)
            node.comparators = [nodes.CoercionNode(node.comparators[0],
                                                   Py_uintptr_t)]

        return node

    def single_compare_objects(self, node):
        op = type(node.ops[0])
        if op not in opmap:
            raise error.NumbaError(
                    node, "%s comparisons not yet implemented" % (op,))

        # Build arguments for PyObject_RichCompareBool
        operator = nodes.const(opmap[op], int_)
        args = [node.left, node.comparators[0], operator]

        # Call PyObject_RichCompareBool
        compare = function_util.external_call(self.context,
                                              self.llvm_module,
                                              'PyObject_RichCompareBool',
                                              args=args)

        # Coerce int result to bool
        return nodes.CoercionNode(compare, bool_)

    def visit_Compare(self, node):
        "Reduce cascaded comparisons into single comparisons"

        # Process children
        self.generic_visit(node)

        compare_nodes = []
        comparators = [nodes.CloneableNode(c) for c in node.comparators]

        # Build comparison nodes
        left = node.left
        for op, right in zip(node.ops, comparators):
            node = ast.Compare(left=left, ops=[op], comparators=[right])
            node = nodes.typednode(node, bool_)

            # Handle comparisons specially based on their types
            node = self.single_compare(node)
            compare_nodes.append(node)

            left = right.clone

        # AND the comparisons together
        node = reduce(build_boolop, reversed(compare_nodes))

        return node
