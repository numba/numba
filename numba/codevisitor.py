# Copyright (c) 2012, Siu Kwan Lam
# All rights reserved.
#
# Modifications by Continuum Analytics, Inc

import logging
import ast

import types, dialect

from .pymothoa.util.descriptor import Descriptor, instanceof
from .pymothoa.compiler_errors import *
from . import nodes, visitors

logger = logging.getLogger(__name__)

class CodeGenerationBase(visitors.NumbaVisitor):

    def __init__(self, context, func, ast):
        super(CodeGenerationBase, self).__init__(context, func, ast)
        self._nodes = []

    @property
    def current_node(self):
        return self._nodes[-1]

    def visit(self, node):
        try:
            fn = getattr(self, 'visit_%s' % type(node).__name__)
        except AttributeError as e:
            logger.exception(e)
            logger.error('Unhandled visit to %s', ast.dump(node))
            raise InternalError(node, 'Not yet implemented.')
        else:
            try:
                self._nodes.append(node) # push current node
                return fn(node)
            except TypeError as e:
                logger.exception(e)
                raise InternalError(node, str(e))
            except (NotImplementedError, AssertionError) as e:
                logger.exception(e)
                raise InternalError(node, str(e))
            finally:
                self._nodes.pop() # pop current node

    def visit_FunctionDef(self, node):
        with self.generate_function(node.name) as fndef:
            # arguments
            self.visit(node.args)
            # function body
            if (isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Str)):
                # Python doc string
                logger.info('Ignoring python doc string.')
                statements = node.body[1:]
            else:
                statements = node.body

            for stmt in statements:
                self.visit(stmt)
        # close function

    def visit_arguments(self, node):
        if node.vararg or node.kwarg or node.defaults:
            raise FunctionDeclarationError(
                'Does not support variable/keyword/default arguments.')

        self.generate_function_arguments([arg.id for arg in arguments])

    def generate_function(self, name):
        raise NotImplementedError

    def generate_function_arguments(self, arguments):
        raise NotImplementedError

    def visit_Call(self, node):
        func = self.visit(node.func)
        return self.generate_call(func, self.visitlist(node.args))

    def generate_declare(self,  name, ty):
        raise NotImplementedError

    def visit_Attribute(self, node):
        result = self.visit(node.value)
        if isinstance(node.ctx, ast.Load):
            return self.generate_load_attribute(node, result)
        else:
            self.generate_store_attribute(node, result)

    def visit_Compare(self, node):
        if len(node.ops) != 1:
            raise NotImplementedError('Multiple operators in ast.Compare')

        if len(node.comparators) != 1:
            raise NotImplementedError('Multiple comparators in ast.Compare')

        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        op  = type(node.ops[0])
        return self.generate_compare(op, lhs, rhs)

    def visit_Return(self, node):
        if node.value is not None:
            value = self.visit(node.value)
            self.generate_return(value)
        else:
            self.generate_return(None)

    def generate_return(self, value):
        raise NotImplementedError

    def generate_compare(self, op_class, lhs, rhs):
        raise NotImplementedError

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = type(node.op)
        return self.generate_binop(op, lhs, rhs)

    def generate_binop(self, op_class, lhs, rhs):
        raise NotImplementedError

    def visit_Assign(self, node):
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        return self.generate_assign(value, target)

    def generate_assign(self, from_value, to_target):
        raise NotImplementedError

    def visit_Num(self, node):
        if node.type.is_int:
            return self.generate_constant_int(node.n)
        elif node.type.is_float:
            return self.generate_constant_real(node.n)
        else:
            assert node.type.is_complex
            return self.generate_constant_complex(node.n)

    def generate_constant_int(self, val):
        raise NotImplementedError

    def generate_constant_real(self, val):
        raise NotImplementedError

    def generate_constant_complex(self, val):
        raise NotImplementedError

    def visit_Subscript(self, node):
        if node.slice.variable.type.is_slice:
            raise NotImplementedError("slicing")

            ptr = self.visit(node.value)
            idx = self.visit(node.slice.lower)
            if node.slice.upper or node.slice.step:
                raise NotImplementedError

            if not isinstance(ptr.type, types.GenericUnboundedArray): # only array
                raise NotImplementedError
            if not isinstance(node.ctx, ast.Load): # only at load context
                raise NotImplementedError

            return self.generate_array_slice(ptr, idx, None, None)

        else:
            assert node.slice.variable.type.is_int

            ptr = self.visit(node.value)
            idx = self.visit(node.slice.value)
            if isinstance(ptr.type, types.GenericVector):
                # Access vector element
                if isinstance(node.ctx, ast.Load): # load
                    return self.generate_vector_load_elem(ptr, idx)
                elif isinstance(node.ctx, ast.Store): # store
                    return self.generate_vector_store_elem(ptr, idx)
            elif isinstance(ptr.type, types.GenericUnboundedArray):
                # Access array element
                if isinstance(node.ctx, ast.Load): # load
                    return self.generate_array_load_elem(ptr, idx)
                elif isinstance(node.ctx, ast.Store): # store
                    return self.generate_array_store_elem(ptr, idx)
            else: # Unsupported types
                raise InvalidSubscriptError(node)

    def generate_array_slice(ptr, lower, upper=None, step=None):
        raise NotImplementedError

    def generate_vector_load_elem(self, ptr, idx):
        raise NotImplementedError

    def generate_vector_store_elem(self, ptr, idx):
        raise NotImplementedError

    def generate_array_load_elem(self, ptr, idx):
        raise NotImplementedError

    def generate_array_store_elem(self, ptr, idx):
        raise NotImplementedError

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load): # load
            try: # lookup in the symbol table
                val = self.symbols[node.id]
            except KeyError: # does not exist
                raise UndefinedSymbolError(node)
            else: # load from stack
                if isinstance(val, int) or isinstance(val, long):
                    return self.generate_constant_int(val)
                elif isinstance(val, float):
                    return self.generate_constant_real(val)
                else:
                    return val
        elif isinstance(node.ctx, ast.Store): # store
            try:
                return self.symbols[node.id]
            except KeyError:
                raise UndefinedSymbolError(node)
        # unreachable
        raise AssertionError('unreachable')

    def visit_If(self, node):
        test = self.visit(node.test)
        iftrue_body = node.body
        orelse_body = node.orelse
        if len(orelse_body) not in [0,1]: raise AssertionError
        self.generate_if(test, iftrue_body, orelse_body)

    def visit_For(self, node):
        if node.orelse:
            raise NotImplementedError('Else in for-loop is not implemented.')
        iternode = node.iter

        str_only_support_forrange = 'Only for-range|for-xrange are supported.'
        if not isinstance(iternode, ast.Call):
            raise InvalidUseOfConstruct(str_only_support_forrange)

        looptype = iternode.func.id
        if looptype not in ['range', 'xrange']:
            raise InvalidUseOfConstruct(str_only_support_forrange)

        # counter variable
        counter_name = node.target.id
        if counter_name in self.symbols:
            raise VariableRedeclarationError(node.target)

        counter_ptr = self.generate_declare(node.target.id, types.Int)
        self.symbols[counter_name] = counter_ptr

        # range information
        iternode_arg_N = len(iternode.args)
        if iternode_arg_N==1: # only END is given
            zero = self.generate_constant_int(0)
            initcount = zero # init count is implicitly zero
            endcountpos = 0
            step = self.generate_constant_int(1)
        elif iternode_arg_N==2: # both BEGIN and END are given
            initcount = self.visit(iternode.args[0]) # init count is given
            endcountpos = 1
            step = self.generate_constant_int(1)
        else: # with BEGIN, END and STEP
            initcount = self.visit(iternode.args[0]) # init count is given
            endcountpos = 1
            step = self.visit(iternode.args[2]) # step is given

        endcount = self.visit(iternode.args[endcountpos]) # end count

        loopbody = node.body
        self.generate_for_range(counter_ptr, initcount, endcount, step, loopbody)

    def generate_for_range(self, counter, init, end, step, body):
        raise NotImplementedError

    def visit_BoolOp(self, node):
        if len(node.values)!=2: raise AssertionError
        return self.generate_boolop(node.op, node.values[0], node.values[1])

    def generate_boolop(self, op_class, lhs, rhs):
        raise NotImplementedError

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return self.generate_not(operand)
        raise NotImplementedError(ast.dump(node))

    def generate_not(self, operand):
        raise NotImplementedError

    def visit_AugAssign(self, node):
        target = self.visit(node.target)
        node.target.ctx = ast.Load() # change context to load
        target_val = self.visit(node.target)
        value = self.visit(node.value)

        result = self.generate_binop(type(node.op), target_val, value)
        return self.generate_assign(result, target)


    def visit_While(self, node):
        if node.orelse:
            raise NotImplementedError('Else in for-loop is not implemented.')
        self.generate_while(node.test, node.body)

    def generate_while(self, test, body):
        raise NotImplementedError


