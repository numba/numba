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

    def op_BINARY_ADD (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_ADD")

    def op_BINARY_AND (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_AND")

    def op_BINARY_DIVIDE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_DIVIDE")

    def op_BINARY_FLOOR_DIVIDE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_FLOOR_DIVIDE")

    def op_BINARY_LSHIFT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_LSHIFT")

    def op_BINARY_MODULO (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_MODULO")

    def op_BINARY_MULTIPLY (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_MULTIPLY")

    def op_BINARY_OR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_OR")

    def op_BINARY_POWER (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_POWER")

    def op_BINARY_RSHIFT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_RSHIFT")

    def op_BINARY_SUBSCR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_SUBSCR")

    def op_BINARY_SUBTRACT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_SUBTRACT")

    def op_BINARY_TRUE_DIVIDE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_TRUE_DIVIDE")

    def op_BINARY_XOR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BINARY_XOR")

    def op_BREAK_LOOP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BREAK_LOOP")

    def op_BUILD_CLASS (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BUILD_CLASS")

    def op_BUILD_LIST (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BUILD_LIST")

    def op_BUILD_MAP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BUILD_MAP")

    def op_BUILD_SET (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BUILD_SET")

    def op_BUILD_SLICE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BUILD_SLICE")

    def op_BUILD_TUPLE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_BUILD_TUPLE")

    def op_CALL_FUNCTION (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_CALL_FUNCTION")

    def op_CALL_FUNCTION_KW (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_CALL_FUNCTION_KW")

    def op_CALL_FUNCTION_VAR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_CALL_FUNCTION_VAR")

    def op_CALL_FUNCTION_VAR_KW (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_CALL_FUNCTION_VAR_KW")

    def op_COMPARE_OP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_COMPARE_OP")

    def op_CONTINUE_LOOP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_CONTINUE_LOOP")

    def op_DELETE_ATTR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DELETE_ATTR")

    def op_DELETE_DEREF (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DELETE_DEREF")

    def op_DELETE_FAST (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DELETE_FAST")

    def op_DELETE_GLOBAL (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DELETE_GLOBAL")

    def op_DELETE_NAME (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DELETE_NAME")

    def op_DELETE_SLICE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DELETE_SLICE")

    def op_DELETE_SUBSCR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DELETE_SUBSCR")

    def op_DUP_TOP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DUP_TOP")

    def op_DUP_TOPX (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DUP_TOPX")

    def op_DUP_TOP_TWO (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_DUP_TOP_TWO")

    def op_END_FINALLY (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_END_FINALLY")

    def op_EXEC_STMT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_EXEC_STMT")

    def op_EXTENDED_ARG (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_EXTENDED_ARG")

    def op_FOR_ITER (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_FOR_ITER")

    def op_GET_ITER (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_GET_ITER")

    def op_IMPORT_FROM (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_IMPORT_FROM")

    def op_IMPORT_NAME (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_IMPORT_NAME")

    def op_IMPORT_STAR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_IMPORT_STAR")

    def op_INPLACE_ADD (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_ADD")

    def op_INPLACE_AND (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_AND")

    def op_INPLACE_DIVIDE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_DIVIDE")

    def op_INPLACE_FLOOR_DIVIDE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_FLOOR_DIVIDE")

    def op_INPLACE_LSHIFT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_LSHIFT")

    def op_INPLACE_MODULO (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_MODULO")

    def op_INPLACE_MULTIPLY (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_MULTIPLY")

    def op_INPLACE_OR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_OR")

    def op_INPLACE_POWER (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_POWER")

    def op_INPLACE_RSHIFT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_RSHIFT")

    def op_INPLACE_SUBTRACT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_SUBTRACT")

    def op_INPLACE_TRUE_DIVIDE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_TRUE_DIVIDE")

    def op_INPLACE_XOR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_INPLACE_XOR")

    def op_JUMP_ABSOLUTE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_JUMP_ABSOLUTE")

    def op_JUMP_FORWARD (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_JUMP_FORWARD")

    def op_JUMP_IF_FALSE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_JUMP_IF_FALSE")

    def op_JUMP_IF_FALSE_OR_POP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_JUMP_IF_FALSE_OR_POP")

    def op_JUMP_IF_TRUE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_JUMP_IF_TRUE")

    def op_JUMP_IF_TRUE_OR_POP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_JUMP_IF_TRUE_OR_POP")

    def op_LIST_APPEND (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LIST_APPEND")

    def op_LOAD_ATTR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LOAD_ATTR")

    def op_LOAD_BUILD_CLASS (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LOAD_BUILD_CLASS")

    def op_LOAD_CLOSURE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LOAD_CLOSURE")

    def op_LOAD_CONST (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LOAD_CONST")

    def op_LOAD_DEREF (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LOAD_DEREF")

    def op_LOAD_FAST (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LOAD_FAST")

    def op_LOAD_GLOBAL (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LOAD_GLOBAL")

    def op_LOAD_LOCALS (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LOAD_LOCALS")

    def op_LOAD_NAME (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_LOAD_NAME")

    def op_MAKE_CLOSURE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_MAKE_CLOSURE")

    def op_MAKE_FUNCTION (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_MAKE_FUNCTION")

    def op_MAP_ADD (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_MAP_ADD")

    def op_NOP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_NOP")

    def op_POP_BLOCK (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_POP_BLOCK")

    def op_POP_EXCEPT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_POP_EXCEPT")

    def op_POP_JUMP_IF_FALSE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_POP_JUMP_IF_FALSE")

    def op_POP_JUMP_IF_TRUE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_POP_JUMP_IF_TRUE")

    def op_POP_TOP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_POP_TOP")

    def op_PRINT_EXPR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_PRINT_EXPR")

    def op_PRINT_ITEM (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_PRINT_ITEM")

    def op_PRINT_ITEM_TO (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_PRINT_ITEM_TO")

    def op_PRINT_NEWLINE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_PRINT_NEWLINE")

    def op_PRINT_NEWLINE_TO (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_PRINT_NEWLINE_TO")

    def op_RAISE_VARARGS (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_RAISE_VARARGS")

    def op_RETURN_VALUE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_RETURN_VALUE")

    def op_ROT_FOUR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_ROT_FOUR")

    def op_ROT_THREE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_ROT_THREE")

    def op_ROT_TWO (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_ROT_TWO")

    def op_SETUP_EXCEPT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_SETUP_EXCEPT")

    def op_SETUP_FINALLY (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_SETUP_FINALLY")

    def op_SETUP_LOOP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_SETUP_LOOP")

    def op_SETUP_WITH (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_SETUP_WITH")

    def op_SET_ADD (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_SET_ADD")

    def op_SLICE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_SLICE")

    def op_STOP_CODE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STOP_CODE")

    def op_STORE_ATTR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STORE_ATTR")

    def op_STORE_DEREF (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STORE_DEREF")

    def op_STORE_FAST (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STORE_FAST")

    def op_STORE_GLOBAL (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STORE_GLOBAL")

    def op_STORE_LOCALS (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STORE_LOCALS")

    def op_STORE_MAP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STORE_MAP")

    def op_STORE_NAME (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STORE_NAME")

    def op_STORE_SLICE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STORE_SLICE")

    def op_STORE_SUBSCR (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_STORE_SUBSCR")

    def op_UNARY_CONVERT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_UNARY_CONVERT")

    def op_UNARY_INVERT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_UNARY_INVERT")

    def op_UNARY_NEGATIVE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_UNARY_NEGATIVE")

    def op_UNARY_NOT (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_UNARY_NOT")

    def op_UNARY_POSITIVE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_UNARY_POSITIVE")

    def op_UNPACK_EX (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_UNPACK_EX")

    def op_UNPACK_SEQUENCE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_UNPACK_SEQUENCE")

    def op_WITH_CLEANUP (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_WITH_CLEANUP")

    def op_YIELD_VALUE (self, i, op, arg, *args, **kws):
        raise NotImplementedError("BytecodeVisitor.op_YIELD_VALUE")

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
