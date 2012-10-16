#! /usr/bin/env python
# ______________________________________________________________________

import dis
import opcode

from bytecode_visitor import BytecodeIterVisitor
import opcode_util

# ______________________________________________________________________

class BytecodeFlowBuilder (BytecodeIterVisitor):
    '''Transforms a bytecode vector into a bytecode "flow tree".

    The flow tree is a Python dictionary, described loosely by the
    following set of productions:

      * `flow_tree` ``:=`` ``{`` `blocks` ``*`` ``}``
      * `blocks` ``:=`` `block_index` ``:`` ``[`` `bytecode_tree` ``*`` ``]``
      * `bytecode_tree` ``:=`` ``(`` `opcode_index` ``,`` `opcode` ``,``
          `opname` ``,`` `arg` ``,`` ``[`` `bytecode_tree` ``*`` ``]`` ``)``

    The primary purpose of this transformation is to simulate the
    value stack, removing it and any stack-specific opcodes.'''

    def __init__ (self, *args, **kws):
        super(BytecodeFlowBuilder, self).__init__(*args, **kws)
        om_items = opcode_util.OPCODE_MAP.items()
        self.opmap = dict((opcode.opmap[opname], (opname, pops, pushes, stmt))
                          for opname, (pops, pushes, stmt) in om_items
                          if opname in opcode.opmap)

    def _visit_op (self, i, op, arg, opname, pops, pushes, appends):
        assert pops is not None, ('%s not well defined in opcode_util.'
                                  'OPCODE_MAP' % opname)
        if pops:
            if pops < 0:
                pops = arg - pops - 1
            stk_args = self.stack[-pops:]
            del self.stack[-pops:]
        else:
            stk_args = []
        ret_val = (i, op, opname, arg, stk_args)
        if pushes:
            self.stack.append(ret_val)
        if appends:
            self.block.append(ret_val)
        return ret_val

    def _op (self, i, op, arg):
        opname, pops, pushes, appends = self.opmap[op]
        return self._visit_op(i, op, arg, opname, pops, pushes, appends)

    def enter_code_object (self, co_obj):
        labels = dis.findlabels(co_obj.co_code)
        labels = opcode_util.extendlabels(co_obj.co_code, labels)
        self.blocks = dict((index, [])
                           for index in labels)
        self.stack = []
        self.loop_stack = []
        self.blocks[0] = self.block = []

    def exit_code_object (self, co_obj):
        ret_val = self.blocks
        del self.stack
        del self.loop_stack
        del self.block
        del self.blocks
        return ret_val

    def visit_op (self, i, op, arg):
        if i in self.blocks:
            self.block = self.blocks[i]
        return super(BytecodeFlowBuilder, self).visit_op(i, op, arg)

    op_BINARY_ADD = _op
    op_BINARY_AND = _op
    op_BINARY_DIVIDE = _op
    op_BINARY_FLOOR_DIVIDE = _op
    op_BINARY_LSHIFT = _op
    op_BINARY_MODULO = _op
    op_BINARY_MULTIPLY = _op
    op_BINARY_OR = _op
    op_BINARY_POWER = _op
    op_BINARY_RSHIFT = _op
    op_BINARY_SUBSCR = _op
    op_BINARY_SUBTRACT = _op
    op_BINARY_TRUE_DIVIDE = _op
    op_BINARY_XOR = _op

    def op_BREAK_LOOP (self, i, op, arg):
        loop_i, _, loop_arg = self.loop_stack[-1]
        assert arg is None
        return self._op(i, op, loop_i + loop_arg + 3)

    #op_BUILD_CLASS = _op
    op_BUILD_LIST = _op
    op_BUILD_MAP = _op
    op_BUILD_SLICE = _op
    op_BUILD_TUPLE = _op
    op_CALL_FUNCTION = _op
    op_CALL_FUNCTION_KW = _op
    op_CALL_FUNCTION_VAR = _op
    op_CALL_FUNCTION_VAR_KW = _op
    op_COMPARE_OP = _op
    #op_CONTINUE_LOOP = _op
    op_DELETE_ATTR = _op
    op_DELETE_FAST = _op
    op_DELETE_GLOBAL = _op
    op_DELETE_NAME = _op
    op_DELETE_SLICE = _op
    op_DELETE_SUBSCR = _op

    def op_DUP_TOP (self, i, op, arg):
        self.stack.append(self.stack[-1])

    def op_DUP_TOPX (self, i, op, arg):
        self.stack += self.stack[-arg:]

    #op_END_FINALLY = _op
    op_EXEC_STMT = _op
    #op_EXTENDED_ARG = _op
    op_FOR_ITER = _op
    op_GET_ITER = _op
    op_IMPORT_FROM = _op
    op_IMPORT_NAME = _op
    op_IMPORT_STAR = _op
    op_INPLACE_ADD = _op
    op_INPLACE_AND = _op
    op_INPLACE_DIVIDE = _op
    op_INPLACE_FLOOR_DIVIDE = _op
    op_INPLACE_LSHIFT = _op
    op_INPLACE_MODULO = _op
    op_INPLACE_MULTIPLY = _op
    op_INPLACE_OR = _op
    op_INPLACE_POWER = _op
    op_INPLACE_RSHIFT = _op
    op_INPLACE_SUBTRACT = _op
    op_INPLACE_TRUE_DIVIDE = _op
    op_INPLACE_XOR = _op
    op_JUMP_ABSOLUTE = _op
    op_JUMP_FORWARD = _op
    op_JUMP_IF_FALSE = _op
    op_JUMP_IF_TRUE = _op
    op_LIST_APPEND = _op
    op_LOAD_ATTR = _op
    op_LOAD_CLOSURE = _op
    op_LOAD_CONST = _op
    op_LOAD_DEREF = _op
    op_LOAD_FAST = _op
    op_LOAD_GLOBAL = _op
    op_LOAD_LOCALS = _op
    op_LOAD_NAME = _op
    op_MAKE_CLOSURE = _op
    op_MAKE_FUNCTION = _op
    op_NOP = _op

    def op_POP_BLOCK (self, i, op, arg):
        self.loop_stack.pop()
        return self._op(i, op, arg)

    op_POP_JUMP_IF_FALSE = _op
    op_POP_JUMP_IF_TRUE = _op
    op_POP_TOP = _op
    op_PRINT_EXPR = _op
    op_PRINT_ITEM = _op
    op_PRINT_ITEM_TO = _op
    op_PRINT_NEWLINE = _op
    op_PRINT_NEWLINE_TO = _op
    op_RAISE_VARARGS = _op
    op_RETURN_VALUE = _op

    def op_ROT_FOUR (self, i, op, arg):
        self.stack[-4:] = (self.stack[-1], self.stack[-4], self.stack[-3],
                           self.stack[-2])

    def op_ROT_THREE (self, i, op, arg):
        self.stack[-3:] = (self.stack[-1], self.stack[-3], self.stack[-2])

    def op_ROT_TWO (self, i, op, arg):
        self.stack[-2:] = (self.stack[-1], self.stack[-2])

    #op_SETUP_EXCEPT = _op
    #op_SETUP_FINALLY = _op

    def op_SETUP_LOOP (self, i, op, arg):
        self.loop_stack.append((i, op, arg))
        self.block.append((i, op, self.opnames[op], arg, []))

    op_SLICE = _op
    #op_STOP_CODE = _op
    op_STORE_ATTR = _op
    op_STORE_DEREF = _op
    op_STORE_FAST = _op
    op_STORE_GLOBAL = _op
    op_STORE_MAP = _op
    op_STORE_NAME = _op
    op_STORE_SLICE = _op
    op_STORE_SUBSCR = _op
    op_UNARY_CONVERT = _op
    op_UNARY_INVERT = _op
    op_UNARY_NEGATIVE = _op
    op_UNARY_NOT = _op
    op_UNARY_POSITIVE = _op
    op_UNPACK_SEQUENCE = _op
    #op_WITH_CLEANUP = _op
    op_YIELD_VALUE = _op

# ______________________________________________________________________

def build_flow (func):
    '''Given a Python function, return a bytecode flow tree for that
    function.'''
    return BytecodeFlowBuilder().visit(opcode_util.get_code_object(func))

# ______________________________________________________________________
# Main (self-test) routine

def main (*args):
    import pprint
    from tests import llfuncs
    if not args:
        args = ('doslice',)
    for arg in args:
        pprint.pprint(build_flow(getattr(llfuncs, arg)))

# ______________________________________________________________________

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of byte_flow.py
