#! /usr/bin/env python
# ______________________________________________________________________

import itertools

import opcode
from opcode_util import itercode

# ______________________________________________________________________

class BytecodeVisitor (object):
    opnames = [name.split('+')[0] for name in opcode.opname]

    def visit_op (self, i, op, arg, *args, **kws):
        if op < 0:
            ret_val = self.visit_synthetic_op(i, op, arg, *args, **kws)
        else:
            method = getattr(self, 'op_' + self.opnames[op])
            ret_val = method(i, op, arg, *args, **kws)
        return ret_val

    def visit_synthetic_op (self, i, op, arg, *args, **kws):
        raise NotImplementedError(
            'BytecodeVisitor.visit_synthetic_op() must be overloaded if using '
            'synthetic opcodes.')

    def _not_implemented (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_%s (@ bytecode index %d)"
                                  % (self.opnames[op], i))

    op_BINARY_ADD = _not_implemented
    op_BINARY_AND = _not_implemented
    op_BINARY_DIVIDE = _not_implemented
    op_BINARY_FLOOR_DIVIDE = _not_implemented
    op_BINARY_LSHIFT = _not_implemented
    op_BINARY_MODULO = _not_implemented
    op_BINARY_MULTIPLY = _not_implemented
    op_BINARY_OR = _not_implemented
    op_BINARY_POWER = _not_implemented
    op_BINARY_RSHIFT = _not_implemented
    op_BINARY_SUBSCR = _not_implemented
    op_BINARY_SUBTRACT = _not_implemented
    op_BINARY_TRUE_DIVIDE = _not_implemented
    op_BINARY_XOR = _not_implemented
    op_BREAK_LOOP = _not_implemented
    op_BUILD_CLASS = _not_implemented
    op_BUILD_LIST = _not_implemented
    op_BUILD_MAP = _not_implemented
    op_BUILD_SET = _not_implemented
    op_BUILD_SLICE = _not_implemented
    op_BUILD_TUPLE = _not_implemented
    op_CALL_FUNCTION = _not_implemented
    op_CALL_FUNCTION_KW = _not_implemented
    op_CALL_FUNCTION_VAR = _not_implemented
    op_CALL_FUNCTION_VAR_KW = _not_implemented
    op_COMPARE_OP = _not_implemented
    op_CONTINUE_LOOP = _not_implemented
    op_DELETE_ATTR = _not_implemented
    op_DELETE_DEREF = _not_implemented
    op_DELETE_FAST = _not_implemented
    op_DELETE_GLOBAL = _not_implemented
    op_DELETE_NAME = _not_implemented
    op_DELETE_SLICE = _not_implemented
    op_DELETE_SUBSCR = _not_implemented
    op_DUP_TOP = _not_implemented
    op_DUP_TOPX = _not_implemented
    op_DUP_TOP_TWO = _not_implemented
    op_END_FINALLY = _not_implemented
    op_EXEC_STMT = _not_implemented
    op_EXTENDED_ARG = _not_implemented
    op_FOR_ITER = _not_implemented
    op_GET_ITER = _not_implemented
    op_IMPORT_FROM = _not_implemented
    op_IMPORT_NAME = _not_implemented
    op_IMPORT_STAR = _not_implemented
    op_INPLACE_ADD = _not_implemented
    op_INPLACE_AND = _not_implemented
    op_INPLACE_DIVIDE = _not_implemented
    op_INPLACE_FLOOR_DIVIDE = _not_implemented
    op_INPLACE_LSHIFT = _not_implemented
    op_INPLACE_MODULO = _not_implemented
    op_INPLACE_MULTIPLY = _not_implemented
    op_INPLACE_OR = _not_implemented
    op_INPLACE_POWER = _not_implemented
    op_INPLACE_RSHIFT = _not_implemented
    op_INPLACE_SUBTRACT = _not_implemented
    op_INPLACE_TRUE_DIVIDE = _not_implemented
    op_INPLACE_XOR = _not_implemented
    op_JUMP_ABSOLUTE = _not_implemented
    op_JUMP_FORWARD = _not_implemented
    op_JUMP_IF_FALSE = _not_implemented
    op_JUMP_IF_FALSE_OR_POP = _not_implemented
    op_JUMP_IF_TRUE = _not_implemented
    op_JUMP_IF_TRUE_OR_POP = _not_implemented
    op_LIST_APPEND = _not_implemented
    op_LOAD_ATTR = _not_implemented
    op_LOAD_BUILD_CLASS = _not_implemented
    op_LOAD_CLOSURE = _not_implemented
    op_LOAD_CONST = _not_implemented
    op_LOAD_DEREF = _not_implemented
    op_LOAD_FAST = _not_implemented
    op_LOAD_GLOBAL = _not_implemented
    op_LOAD_LOCALS = _not_implemented
    op_LOAD_NAME = _not_implemented
    op_MAKE_CLOSURE = _not_implemented
    op_MAKE_FUNCTION = _not_implemented
    op_MAP_ADD = _not_implemented
    op_NOP = _not_implemented
    op_POP_BLOCK = _not_implemented
    op_POP_EXCEPT = _not_implemented
    op_POP_JUMP_IF_FALSE = _not_implemented
    op_POP_JUMP_IF_TRUE = _not_implemented
    op_POP_TOP = _not_implemented
    op_PRINT_EXPR = _not_implemented
    op_PRINT_ITEM = _not_implemented
    op_PRINT_ITEM_TO = _not_implemented
    op_PRINT_NEWLINE = _not_implemented
    op_PRINT_NEWLINE_TO = _not_implemented
    op_RAISE_VARARGS = _not_implemented
    op_RETURN_VALUE = _not_implemented
    op_ROT_FOUR = _not_implemented
    op_ROT_THREE = _not_implemented
    op_ROT_TWO = _not_implemented
    op_SETUP_EXCEPT = _not_implemented
    op_SETUP_FINALLY = _not_implemented
    op_SETUP_LOOP = _not_implemented
    op_SETUP_WITH = _not_implemented
    op_SET_ADD = _not_implemented
    op_SLICE = _not_implemented
    op_STOP_CODE = _not_implemented
    op_STORE_ATTR = _not_implemented
    op_STORE_DEREF = _not_implemented
    op_STORE_FAST = _not_implemented
    op_STORE_GLOBAL = _not_implemented
    op_STORE_LOCALS = _not_implemented
    op_STORE_MAP = _not_implemented
    op_STORE_NAME = _not_implemented
    op_STORE_SLICE = _not_implemented
    op_STORE_SUBSCR = _not_implemented
    op_UNARY_CONVERT = _not_implemented
    op_UNARY_INVERT = _not_implemented
    op_UNARY_NEGATIVE = _not_implemented
    op_UNARY_NOT = _not_implemented
    op_UNARY_POSITIVE = _not_implemented
    op_UNPACK_EX = _not_implemented
    op_UNPACK_SEQUENCE = _not_implemented
    op_WITH_CLEANUP = _not_implemented
    op_YIELD_VALUE = _not_implemented

# ______________________________________________________________________

class BytecodeIterVisitor (BytecodeVisitor):
    def visit (self, co_obj):
        self.enter_code_object(co_obj)
        for i, op, arg in itercode(co_obj.co_code):
            self.visit_op(i, op, arg)
        return self.exit_code_object(co_obj)

    def enter_code_object (self, co_obj):
        pass

    def exit_code_object (self, co_obj):
        pass

# ______________________________________________________________________

class BytecodeFlowVisitor (BytecodeVisitor):
    def visit (self, flow):
        self.block_list = list(flow.keys())
        self.block_list.sort()
        self.enter_flow_object(flow)
        for block in self.block_list:
            prelude = self.enter_block(block)
            prelude_isa_list = isinstance(prelude, list)
            if prelude or prelude_isa_list:
                if not prelude_isa_list:
                    prelude = []
                new_stmts = list(self.visit_op(i, op, arg, *args)
                                 for i, op, _, arg, args in flow[block])
                self.new_flow[block] = list(itertools.chain(
                    prelude, *new_stmts))
            self.exit_block(block)
        del self.block_list
        return self.exit_flow_object(flow)

    def visit_op (self, i, op, arg, *args, **kws):
        new_args = []
        for child_i, child_op, _, child_arg, child_args in args:
            new_args.extend(self.visit_op(child_i, child_op, child_arg,
                                          *child_args))
        ret_val = super(BytecodeFlowVisitor, self).visit_op(i, op, arg,
                                                            *new_args)
        return ret_val

    def enter_flow_object (self, flow):
        self.new_flow = {}

    def exit_flow_object (self, flow):
        ret_val = self.new_flow
        del self.new_flow
        return ret_val

    def enter_block (self, block):
        pass

    def exit_block (self, block):
        pass

# ______________________________________________________________________

class BenignBytecodeVisitorMixin (object):
    def _do_nothing (self, i, op, arg, *args, **kws):
        return [(i, op, self.opnames[op], arg, args)]

    op_BINARY_ADD = _do_nothing
    op_BINARY_AND = _do_nothing
    op_BINARY_DIVIDE = _do_nothing
    op_BINARY_FLOOR_DIVIDE = _do_nothing
    op_BINARY_LSHIFT = _do_nothing
    op_BINARY_MODULO = _do_nothing
    op_BINARY_MULTIPLY = _do_nothing
    op_BINARY_OR = _do_nothing
    op_BINARY_POWER = _do_nothing
    op_BINARY_RSHIFT = _do_nothing
    op_BINARY_SUBSCR = _do_nothing
    op_BINARY_SUBTRACT = _do_nothing
    op_BINARY_TRUE_DIVIDE = _do_nothing
    op_BINARY_XOR = _do_nothing
    op_BREAK_LOOP = _do_nothing
    op_BUILD_CLASS = _do_nothing
    op_BUILD_LIST = _do_nothing
    op_BUILD_MAP = _do_nothing
    op_BUILD_SET = _do_nothing
    op_BUILD_SLICE = _do_nothing
    op_BUILD_TUPLE = _do_nothing
    op_CALL_FUNCTION = _do_nothing
    op_CALL_FUNCTION_KW = _do_nothing
    op_CALL_FUNCTION_VAR = _do_nothing
    op_CALL_FUNCTION_VAR_KW = _do_nothing
    op_COMPARE_OP = _do_nothing
    op_CONTINUE_LOOP = _do_nothing
    op_DELETE_ATTR = _do_nothing
    op_DELETE_DEREF = _do_nothing
    op_DELETE_FAST = _do_nothing
    op_DELETE_GLOBAL = _do_nothing
    op_DELETE_NAME = _do_nothing
    op_DELETE_SLICE = _do_nothing
    op_DELETE_SUBSCR = _do_nothing
    op_DUP_TOP = _do_nothing
    op_DUP_TOPX = _do_nothing
    op_DUP_TOP_TWO = _do_nothing
    op_END_FINALLY = _do_nothing
    op_EXEC_STMT = _do_nothing
    op_EXTENDED_ARG = _do_nothing
    op_FOR_ITER = _do_nothing
    op_GET_ITER = _do_nothing
    op_IMPORT_FROM = _do_nothing
    op_IMPORT_NAME = _do_nothing
    op_IMPORT_STAR = _do_nothing
    op_INPLACE_ADD = _do_nothing
    op_INPLACE_AND = _do_nothing
    op_INPLACE_DIVIDE = _do_nothing
    op_INPLACE_FLOOR_DIVIDE = _do_nothing
    op_INPLACE_LSHIFT = _do_nothing
    op_INPLACE_MODULO = _do_nothing
    op_INPLACE_MULTIPLY = _do_nothing
    op_INPLACE_OR = _do_nothing
    op_INPLACE_POWER = _do_nothing
    op_INPLACE_RSHIFT = _do_nothing
    op_INPLACE_SUBTRACT = _do_nothing
    op_INPLACE_TRUE_DIVIDE = _do_nothing
    op_INPLACE_XOR = _do_nothing
    op_JUMP_ABSOLUTE = _do_nothing
    op_JUMP_FORWARD = _do_nothing
    op_JUMP_IF_FALSE = _do_nothing
    op_JUMP_IF_FALSE_OR_POP = _do_nothing
    op_JUMP_IF_TRUE = _do_nothing
    op_JUMP_IF_TRUE_OR_POP = _do_nothing
    op_LIST_APPEND = _do_nothing
    op_LOAD_ATTR = _do_nothing
    op_LOAD_BUILD_CLASS = _do_nothing
    op_LOAD_CLOSURE = _do_nothing
    op_LOAD_CONST = _do_nothing
    op_LOAD_DEREF = _do_nothing
    op_LOAD_FAST = _do_nothing
    op_LOAD_GLOBAL = _do_nothing
    op_LOAD_LOCALS = _do_nothing
    op_LOAD_NAME = _do_nothing
    op_MAKE_CLOSURE = _do_nothing
    op_MAKE_FUNCTION = _do_nothing
    op_MAP_ADD = _do_nothing
    op_NOP = _do_nothing
    op_POP_BLOCK = _do_nothing
    op_POP_EXCEPT = _do_nothing
    op_POP_JUMP_IF_FALSE = _do_nothing
    op_POP_JUMP_IF_TRUE = _do_nothing
    op_POP_TOP = _do_nothing
    op_PRINT_EXPR = _do_nothing
    op_PRINT_ITEM = _do_nothing
    op_PRINT_ITEM_TO = _do_nothing
    op_PRINT_NEWLINE = _do_nothing
    op_PRINT_NEWLINE_TO = _do_nothing
    op_RAISE_VARARGS = _do_nothing
    op_RETURN_VALUE = _do_nothing
    op_ROT_FOUR = _do_nothing
    op_ROT_THREE = _do_nothing
    op_ROT_TWO = _do_nothing
    op_SETUP_EXCEPT = _do_nothing
    op_SETUP_FINALLY = _do_nothing
    op_SETUP_LOOP = _do_nothing
    op_SETUP_WITH = _do_nothing
    op_SET_ADD = _do_nothing
    op_SLICE = _do_nothing
    op_STOP_CODE = _do_nothing
    op_STORE_ATTR = _do_nothing
    op_STORE_DEREF = _do_nothing
    op_STORE_FAST = _do_nothing
    op_STORE_GLOBAL = _do_nothing
    op_STORE_LOCALS = _do_nothing
    op_STORE_MAP = _do_nothing
    op_STORE_NAME = _do_nothing
    op_STORE_SLICE = _do_nothing
    op_STORE_SUBSCR = _do_nothing
    op_UNARY_CONVERT = _do_nothing
    op_UNARY_INVERT = _do_nothing
    op_UNARY_NEGATIVE = _do_nothing
    op_UNARY_NOT = _do_nothing
    op_UNARY_POSITIVE = _do_nothing
    op_UNPACK_EX = _do_nothing
    op_UNPACK_SEQUENCE = _do_nothing
    op_WITH_CLEANUP = _do_nothing
    op_YIELD_VALUE = _do_nothing

# ______________________________________________________________________
# End of bytecode_visitor.py
