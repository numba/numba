# -*- encoding: UTF-8 -*-

import minivisitor
import miniutils
import minitypes
import specializers

def admissible(broadcasting_tuple, n_loops):
    """
    Check for admissibility. Indicates whether partial hoisting is the most
    efficient thing to perform. See also partially_hoistable()
    """
    if len(broadcasting_tuple) < n_loops:
        # In this this situation, we pad with leading broadcasting dimensions.
        # This means we have to hoist all the way
        return False

    # Filter leading False values
    i = 0
    for i, broadcasting in enumerate(broadcasting_tuple):
        if broadcasting:
            break

    # Check for all trailing values (at least one) being True
    return broadcasting_tuple[i:] and miniutils.all(broadcasting_tuple[i:])

def partially_hoistable(broadcasting_tuple, n_loops):
    """
    This function indicates, when admissible() returns false, whether an
    expression is partially hoistable. This means the caller must establish
    whether repeated computation or an array temporary will be more beneficial.

    If the expression is a variable, there is no repeaetd computation, and
    it should be hoisted as far as possible.
    """
    return broadcasting_tuple[-1]


def broadcasting(broadcasting_tuple1, broadcasting_tuple2):
    return broadcasting_tuple1 != broadcasting_tuple2

class HoistBroadcastingExpressions(specializers.BaseSpecializer):
    """
    This transform hoists out part of sub-expressions which are broadcasting.
    There are two cases:

        1) We can hoist out the sub-expression and store it in a scalar for
           broadcasting
        2) We have to hoist the sub-expression out entirely and store it in
           a temporary array

    As an alternative to 2), we could swap axes to return to situation 1),
    i.e. reorder the element-wise traversal order. We do not choose this
    option, since the loop order is tailored to cache-friendliness.

    We determine where to hoist sub-expression to based on broadcasting
    information. Each type has a broadcasting tuple with a true/false value
    for each dimension, specifying whether it will broadcast in that dimension
    (i.e. broadcasting is not optional in that dimension).

    We make the following observations:

        1) Trailing truth values mean we can hoist the sub-expression out just
           before the first truth value in the consecutive sequence of truth
           values

            Example ``(False, True, True)``::

                A[:, None, None] * A[:, None, None] * B[:, :, :]

            becomes::

                for i in shape[0]:
                    temp = A[i, 0, 0] * A[i, 0, 0]
                    for j in shape[1]:
                        for k in shape[2]:
                            temp * B[i, j, k]

        2) If not all consecutive leading values are false, we have to assign
           to a temporary array (i.e., hoist out all the way)

            Example ``(True, True, False)``::

                A[None, None, :] * A[None, None, :] * B[:, :, :]

            becomes::

                allocate temp

                for k in shape[2]:
                    temp[k] = A[0, 0, k] * A[0, 0, k]

                for i in shape[0]:
                    for j in shape[1]:
                        for k in shape[2]:
                            temp[k] * B[i, j, k]

                deallocate temp

    More generally, if the index sequence of array A is not an admissible prefix
    of the total index sequence, we have situation 2). For instance,
    ``(True, False, True)`` would mean we could hoist out the expression one
    level, but we would still have repeated computation. What we could do in
    this case, in addition to 2), is reduce indexing overhead, i.e. generate::

        for j in shape[1]:
            temp[j] = A[0, j, 0] * A[0, j, 0]

        for i in shape[0]:
            for j in shape[1]:
                temp_scalar = temp[j]
                for k in shape[2]:
                    temp_scalar * B[i, j, k]

    This is bonus points.
    """

    def visit_FunctionNode(self, node):
        self.function = node
        inner_loop = node.for_loops[-1]
        self.visitchildren(inner_loop)
        return node

    def visit_Variable(self, node):
        type = node.type
        if type.is_array and type.broadcasting is not None:
            n_loops = len(self.function.for_loops)
            if admissible(type.broadcasting, n_loops):
                node.hoistable = True
            elif partially_hoistable(type.broadcasting, n_loops):
                # TODO: see whether `node` should be fully (array temporary)
                # TODO: or partially hoisted
                node.hoistable = True
            elif miniutils.any(type.broadcasting):
                pass # enable when temporaries are implemented in minivect
                # node.need_temp = True

            node.broadcasting = type.broadcasting

        return node

    def visit_ArrayAtribute(self, node):
        return node

    def _hoist_binop_operands(self, b, node):
        # perform hoisitng. Does not handle all expressions correctly yet.
        if not node.lhs.hoistable or not node.rhs.hoistable:
            if node.lhs.hoistable:
                node.lhs = self.hoist(node.lhs)
            else:
                node.rhs = self.hoist(node.rhs)

            return node

        lhs_hoisting_level = self.hoisting_level(node.lhs)
        rhs_hoisting_level = self.hoisting_level(node.rhs)

        if lhs_hoisting_level == rhs_hoisting_level:
            node.hoistable = True
            node.broadcasting = node.lhs.broadcasting
            return node

        def binop():
            result = b.binop(node.type, node.operator, node.lhs, node.rhs)
            result.broadcasting = broadcasting
            result.hoistable = True
            return result

        if lhs_hoisting_level < rhs_hoisting_level:
            broadcasting = node.rhs.broadcasting
            node.lhs = self.hoist(node.lhs)
            return self.hoist(binop())
        else: # lhs_hoisting_level > rhs_hoisting_level
            broadcasting = node.lhs.broadcasting
            node.rhs = self.hoist(node.rhs)
            return self.hoist(binop())

    def _make_temp_binop_operands(self, node):
        if broadcasting(node.lhs.broadcasting, node.rhs.broadcasting):
            node.need_temp = True
        else:
            if node.lhs.need_temp:
                node.lhs = self.make_temp(node.lhs)
            if node.rhs.need_temp:
                node.rhs = self.make_temp(node.rhs)

    def visit_BinaryOperationNode(self, node):
        b = self.astbuilder

        self.visitchildren(node)

        node.broadcasting = None

        if node.lhs.need_temp or node.rhs.need_temp:
            return self._make_temp_binop_operands(node)
        elif node.lhs.hoistable or node.rhs.hoistable:
            return self._hoist_binop_operands(b, node)

        return node

    def visit_ForNode(self, node):
        self.visitchildren(node)
        self.handle_pending_stats(node)
        return node

    def visit_BinopNode(self, node):
        # used in superclass, override here
        return self.visit_BinaryOperationNode(node)

    def visit_AssignmentExpr(self, node):
        rhs = self.visit(node.rhs)
        node.rhs = self.process_expr(rhs)
        return node

    def visit_UnopNode(self, node):
        o = node.operand = self.visit(node.operand)

        node.hoistable = o.hoistable
        node.need_temp = o.need_temp
        node.broadcasting = o.broadcasting

        return node

    def process_expr(self, expr):
        if expr.hoistable:
            return self.hoist(expr)
        elif expr.need_temp:
            return self.make_temp(expr)
        else:
            return expr

    def make_temp(self, node):
        "Not implemented yet"
        return node

    def hoisting_level(self, node):
        i = 0
        for i, broadcasting in enumerate(node.broadcasting[::-1]):
            if not broadcasting:
                break

        return self.function.ndim - 1 - i

    def hoist(self, node):
        if not node.hoistable:
            return node

        b = self.astbuilder

        hoisting_level = self.hoisting_level(node)
        if hoisting_level < 0:
            for_loop = self.function
        else:
            for_loop = self.function.for_loops[hoisting_level]

        temp = b.temp(node.type.dtype, name='hoisted_temp')
        temp.broadcasting = None

        # TODO: keep track of the variables
        for variable in self.treepath(node, '//Variable'):
            variable.hoisted = True

        stat = b.assign(temp, node, may_reorder=False)
        for_loop.body = b.stats(stat, for_loop.body)
        return self.visit(temp)
