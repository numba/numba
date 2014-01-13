# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast

from numba import *
from numba import visitors, nodes, error, function_util

logger = logging.getLogger(__name__)

from numba.typesystem import is_obj

#------------------------------------------------------------------------
# 'raise'
#------------------------------------------------------------------------

class LowerRaise(visitors.NumbaTransformer):
    """
    Resolve the 'raise' statement.
    """

    def visit_Raise(self, node):
        # Create void * temporaries
        args = [] # Type, Value, Traceback, Cause
        for arg in [node.type, node.inst, node.tback, None]:
            if arg:
                arg = nodes.CoercionNode(arg, object_)
                arg = nodes.PointerFromObject(arg)
            else:
                arg = nodes.NULL

            args.append(arg)

        # Call numba/external/utitilies/cpyutils.c:do_raise()
        set_exc = function_util.utility_call(
            self.context,
            self.llvm_module,
            'Raise', args)

        result = self.visit(set_exc)
        return result

#------------------------------------------------------------------------
# Specialize Error Checking and Raising
#------------------------------------------------------------------------

class ExceptionSpecializer(visitors.NumbaTransformer):
    """
    Specialize exception handling. Handle error checking and raising.
    """

    def __init__(self, *args, **kwargs):
        super(ExceptionSpecializer, self).__init__(*args, **kwargs)
        self.visited_callnodes = set()

    #------------------------------------------------------------------------
    # Error Checking
    #------------------------------------------------------------------------

    def visit_CheckErrorNode(self, node):
        if node.badval is not None:
            badval = node.badval
            eq = ast.Eq()
        else:
            assert node.goodval is not None
            badval = node.goodval
            eq = ast.NotEq()

        check = nodes.if_else(eq, node.return_value, badval,
                              lhs=node.raise_node, rhs=None)
        return self.visit(check)

    def visit_PyErr_OccurredNode(self, node):
        check_err = nodes.CheckErrorNode(
            nodes.ptrtoint(function_util.external_call(
                self.context,
                self.llvm_module,
                'PyErr_Occurred')),
            goodval=nodes.ptrtoint(nodes.NULL))
        result = nodes.CloneableNode(node.node)
        result = nodes.ExpressionNode(stmts=[result, check_err],
                                      expr=result.clone)
        return self.visit(result)

    #------------------------------------------------------------------------
    # Error Checking for Function Calls
    #------------------------------------------------------------------------

    def visit_NativeCallNode(self, node):
        badval, goodval = node.badval, node.goodval
        if node not in self.visited_callnodes and (badval is not None or
                                                   goodval is not None):
            self.visited_callnodes.add(node)

            result = node.cloneable
            body = nodes.CheckErrorNode(
                result, badval, goodval,
                node.exc_type, node.exc_msg, node.exc_args)

            node = nodes.ExpressionNode(stmts=[body],
                                        expr=result.clone)
            return self.visit(node)
        else:
            self.generic_visit(node)
            return node

    #------------------------------------------------------------------------
    # Check for UnboundLocalError
    #------------------------------------------------------------------------

    def visit_Name(self, node):
        if (is_obj(node.type) and isinstance(node.ctx, ast.Load) and
            getattr(node, 'cf_maybe_null', False)):
            # Check for unbound objects and raise UnboundLocalError if so
            value = nodes.LLVMValueRefNode(Py_uintptr_t, None)
            node.loaded_name = value

            exc_msg = node.variable.name
            if hasattr(node, 'lineno'):
                exc_msg = '%s%s' % (error.format_pos(node), exc_msg)

            check_unbound = nodes.CheckErrorNode(
                value, badval=nodes.const(0, Py_uintptr_t),
                exc_type=UnboundLocalError,
                exc_msg=exc_msg)
            node.check_unbound = self.visit(check_unbound)

        return node

    #------------------------------------------------------------------------
    # Exception Raising
    #------------------------------------------------------------------------

    def _raise_exception(self, node):
        if node.exc_type:
            assert node.exc_msg

            if node.exc_args:
                args = [node.exc_type, node.exc_msg] + node.exc_args
                raise_node = function_util.external_call(self.context,
                                                         self.llvm_module,
                                                         'PyErr_Format',
                                                         args=args)
            else:
                args = [node.exc_type, node.exc_msg]
                raise_node = function_util.external_call(self.context,
                                                         self.llvm_module,
                                                         'PyErr_SetString',
                                                         args=args)

            return [raise_node]
        return []

    def _trap(self, node):
        body = []
        if node.exc_msg and node.print_on_trap:
            pos = error.format_pos(node)
            if node.exception_type:
                exc_type = '%s: ' % node.exception_type.__name__
            else:
                exc_type = ''

            msg = '%s%s%%s' % (exc_type, pos)
            format = nodes.const(msg, c_string_type)
            print_msg = function_util.external_call(self.context,
                                                    self.llvm_module,
                                                    'printf',
                                                    args=[format,
                                                          node.exc_msg])
            body.append(print_msg)

        trap = nodes.LLVMIntrinsicNode(signature=void(), args=[],
                                       func_name='TRAP')
        return body + [trap]

    def visit_RaiseNode(self, node):
        if self.nopython:
            result = self._trap(node)
        else:
            result = self._raise_exception(node)

        return ast.Suite(body=result + [nodes.PropagateNode()])
