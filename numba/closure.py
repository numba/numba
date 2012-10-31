import ast
import math
import cmath
import copy
import opcode
import types
import __builtin__ as builtins

import numba
from numba import *
from numba import error, transforms, closure
from .minivect import minierror, minitypes
from . import translate, utils, _numba_types as numba_types
from .symtab import Variable
from . import visitors, nodes, error
from numba import stdio_util
from numba._numba_types import is_obj, promote_closest
from numba.utils import dump

import llvm.core
import numpy
import numpy as np

import logging
logger = logging.getLogger(__name__)

class ClosureMixin(object):

    function_level = 0

    def _visit_func_children(self, node):
        self.function_level += 1
        self.generic_visit(node)
        self.function_level -= 1
        return node

    def _err_decorator(self, decorator):
        raise error.NumbaError(
                decorator, "Only @jit and @autojit decorators are supported")

    def _check_valid_argtype(self, argtype_node, argtype):
        if not isinstance(argtype, minitypes.Type):
            raise error.NumbaError(argtype_node, "Invalid type: %r" % (argtype,))

    def _assert_constant(self, decorator, result_node):
        result = self.visit(result_node)
        if not result.variable.is_constant:
            raise error.NumbaError(decorator, "Expected a constant")

        return result.variable.constant_value

    def _parse_argtypes(self, decorator, func_def, jit_args):
        argtypes_node = jit_args['argtypes']
        if argtypes_node is None:
            raise error.NumbaError(func_def.args[0],
                                   "Expected an argument type")

        argtypes = self._assert_constant(decorator, argtypes_node)

        if not isinstance(argtypes, (list, tuple)):
            raise error.NumbaError(argtypes_node,
                                   'Invalid argument for argtypes')
        for argtype in argtypes:
            self._check_valid_argtype(argtypes_node, argtype)

        return argtypes

    def _parse_restype(self, decorator, jit_args):
        restype_node = jit_args['restype']

        if restype_node is not None:
            restype = self._assert_constant(decorator, restype_node)
            if isinstance(restype, (string, unicode)):
                restype = decorators._process_sig(restype)

            self._check_valid_argtype(restype_node, restype)
        else:
            restype = None

        return restype

    def _handle_jit_decorator(self, func_def, decorator):
        jit_args = _parse_args(decorator, ['restype', 'argtypes', 'backend',
                                           'target', 'nopython'])

        if decorator.args or decorator.keywords:
            restype = self._parse_restype(decorator, jit_args)
            if restype is not None and restype.is_function:
                signature = restype
            else:
                argtypes = self._parse_argtypes(decorator, func_def, jit_args)
                signature = minitypes.FunctionType(restype, argtypes,
                                                   name=func_def.name)
        else: #elif func_def.args:
            raise error.NumbaError(decorator,
                                   "The argument types and return type "
                                   "need to be specified")
        #else:
        #    signature = minitypes.FunctionType(None, [])

        # TODO: Analyse closure at call or outer function return time to
        # TODO:     infer return type
        # TODO: parse out nopython argument
        return signature

    def _process_decorators(self, node):
        if not node.decorator_list:
            raise error.NumbaError(
                node, "Closure must be decorated with 'jit' or 'autojit'")

        if len(node.decorator_list) > 1:
            raise error.NumbaError(
                        node, "Only one decorator may be specified for "
                              "closure (@jit/@autojit)")

        decorator, = node.decorator_list

        if isinstance(decorator, ast.Name):
            decorator_name = decorator.id
        elif (not isinstance(decorator, ast.Call) or not
                  isinstance(decorator.func, ast.Name)):
            self._err_decorator(decorator)
        else:
            decorator_name = decorator.func.id

        if decorator_name not in ('jit', 'autojit'):
            self._err_decorator(decorator)

        if decorator_name == 'autojit':
            raise error.NumbaError(
                decorator, "Dynamic closures not yet supported, use @jit")

        signature = self._handle_jit_decorator(node, decorator)
        return signature

    def visit_FunctionDef(self, node):
        if self.function_level == 0:
            return self._visit_func_children(node)

        signature = self._process_decorators(node)
        type = numba_types.ClosureType(signature, node)

        self.symtab[node.name] = Variable(type, is_local=True)
        self.ast.closures.append(nodes.ClosureNode(node, type))

        return None


class ClosureResolver(visitors.NumbaVisitor):
    """
    1) run type inference on inner functions
    2) build scope extension type
    3) generate code to instantiate scope extension type at call time
    4) rewrite local variable accesses to accesses on the instantiated scope
    5) when coercing to object, instantiate function with scope as
       first argument
    """

    def __init__(self, *args, **kwargs):
        super(ClosureResolver, self).__init__(*args, **kwargs)
        self.enclosing = [set()]

    def visit_FunctionDef(self, node):
        process_closures(self.context, node, self.symtab)

        cellvars = dict((name, var) for var in self.symtab.iteritems()
                            if var.is_cellvar)
        node.cellvars = cellvars

        fields = [('var%d' % i, cellvar) for i in enumerate(cellvars)]
        if self.parent_scope:
            fields.insert(0, ('base', self.parent_scope.__numba_ext_type))

        class py_class(object):
            pass

        func_name = self.func.__name__
        py_class.__name__ = '%s_scope' % func_name
        ext_type = numba_types.ExtensionType(py_class)

        ext_type.symtab.update(fields)
        ext_type.attribute_struct = numba.struct(fields)

        self.closure_scope_type = ext_type
        self.closure_scope_ext_type = extension_types.create_new_extension_type(
                func_name , (object,), {}, ext_type,
                vtab=None, vtab_type=numba.struct(),
                llvm_methods=[], method_pointers=[])

        if self.parent_scope:
            outer_scope = ast.Name(id='__numba_closure_scope', ctx=ast.Load())
        else:
            outer_scope = None

        cellvar_scope = nodes.InstantiateClosureScope(
                            node.closure_scope_type, outer_scope).cloneable
        node.body.insert(0, cellvar_scope)
        self.cellvar_scope = cellvar_scope.clone

        self.generic_visit(node)
        return node

    #def visit_ClosureNode(self, node):
    #    node.func_def = self.visit(node.func_def)
    #    return node

    def visit_Name(self, node):
        if node.variable.is_cellvar:
            return nodes.ExtTypeAttribute(self.cellvar_scope,
                                          node.id, node.ctx,
                                          self.closure_scope_type)
        elif node.variable.is_freevar:
            pass
        else:
            return node


def get_locals(symtab):
    return dict((name, var.name) for var in symtab.iteritems() if var.is_local)

def process_closures(context, outer_func_def, outer_symtab):
    """
    Process closures recursively and for each variable in each function
    determine whether it is a freevar, a cellvar, a local or otherwise.
    """
    outer_symtab = get_locals(outer_symtab)
    if outer_func_def.closure_scope is not None:
        closure_scope = dict(outer_func_def.closure_scope, **outer_symtab)
    else:
        closure_scope = outer_symtab

    for closure in outer_func_def.closures:
        p, result = pipeline.infer_types_from_ast_and_sig(
                    context, closure.py_func, closure.func_def,
                    closure.type.signature,
                    closure_scope=closure_scope)

        _, symtab, ast = result
        closure.symtab = symtab
        closure.type_inferred_ast = ast

#        if closure.need_wrapper:
#             wrapper, lfunc = p.translator.build_wrapper_function(
#                                                        get_lfunc=True)
#             closure.wrapper_func = wrapper
#             closure.wrapper_lfunc = lfunc