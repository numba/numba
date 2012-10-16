#! /usr/bin/env python
# ______________________________________________________________________

import dis
import opcode

# ______________________________________________________________________
# Module data

hasjump = opcode.hasjrel + opcode.hasjabs
hascbranch = [op for op in hasjump
              if 'IF' in opcode.opname[op]
              or opcode.opname[op] in ('FOR_ITER', 'SETUP_LOOP')]

# Since the actual opcode value may change, manage opcode abstraction
# data by opcode name.

OPCODE_MAP = {
    'BINARY_ADD': (2, 1, None),
    'BINARY_AND': (2, 1, None),
    'BINARY_DIVIDE': (2, 1, None),
    'BINARY_FLOOR_DIVIDE': (2, 1, None),
    'BINARY_LSHIFT': (2, 1, None),
    'BINARY_MODULO': (2, 1, None),
    'BINARY_MULTIPLY': (2, 1, None),
    'BINARY_OR': (2, 1, None),
    'BINARY_POWER': (2, 1, None),
    'BINARY_RSHIFT': (2, 1, None),
    'BINARY_SUBSCR': (2, 1, None),
    'BINARY_SUBTRACT': (2, 1, None),
    'BINARY_TRUE_DIVIDE': (2, 1, None),
    'BINARY_XOR': (2, 1, None),
    'BREAK_LOOP': (0, None, 1),
    'BUILD_CLASS': (None, None, None),
    'BUILD_LIST': (-1, 1, None),
    'BUILD_MAP': (None, None, None),
    'BUILD_SET': (None, None, None),
    'BUILD_SLICE': (None, None, None),
    'BUILD_TUPLE': (-1, 1, None),
    'CALL_FUNCTION': (-2, 1, None),
    'CALL_FUNCTION_KW': (-3, 1, None),
    'CALL_FUNCTION_VAR': (-3, 1, None),
    'CALL_FUNCTION_VAR_KW': (-4, 1, None),
    'COMPARE_OP': (2, 1, None),
    'CONTINUE_LOOP': (None, None, None),
    'DELETE_ATTR': (1, None, 1),
    'DELETE_DEREF': (None, None, None),
    'DELETE_FAST': (0, None, 1),
    'DELETE_GLOBAL': (0, None, 1),
    'DELETE_NAME': (0, None, 1),
    'DELETE_SLICE+0': (1, None, 1),
    'DELETE_SLICE+1': (2, None, 1),
    'DELETE_SLICE+2': (2, None, 1),
    'DELETE_SLICE+3': (3, None, 1),
    'DELETE_SUBSCR': (2, None, 1),
    'DUP_TOP': (None, None, None),
    'DUP_TOPX': (None, None, None),
    'DUP_TOP_TWO': (None, None, None),
    'END_FINALLY': (None, None, None),
    'EXEC_STMT': (None, None, None),
    'EXTENDED_ARG': (None, None, None),
    'FOR_ITER': (1, 1, 1),
    'GET_ITER': (1, 1, None),
    'IMPORT_FROM': (None, None, None),
    'IMPORT_NAME': (None, None, None),
    'IMPORT_STAR': (1, None, 1),
    'INPLACE_ADD': (2, 1, None),
    'INPLACE_AND': (2, 1, None),
    'INPLACE_DIVIDE': (2, 1, None),
    'INPLACE_FLOOR_DIVIDE': (2, 1, None),
    'INPLACE_LSHIFT': (2, 1, None),
    'INPLACE_MODULO': (2, 1, None),
    'INPLACE_MULTIPLY': (2, 1, None),
    'INPLACE_OR': (2, 1, None),
    'INPLACE_POWER': (2, 1, None),
    'INPLACE_RSHIFT': (2, 1, None),
    'INPLACE_SUBTRACT': (2, 1, None),
    'INPLACE_TRUE_DIVIDE': (2, 1, None),
    'INPLACE_XOR': (2, 1, None),
    'JUMP_ABSOLUTE': (0, None, 1),
    'JUMP_FORWARD': (0, None, 1),
    'JUMP_IF_FALSE': (1, None, 1),
    'JUMP_IF_FALSE_OR_POP': (None, None, None),
    'JUMP_IF_TRUE': (1, None, 1),
    'JUMP_IF_TRUE_OR_POP': (None, None, None),
    'LIST_APPEND': (2, 0, 1),
    'LOAD_ATTR': (1, 1, None),
    'LOAD_BUILD_CLASS': (None, None, None),
    'LOAD_CLOSURE': (None, None, None),
    'LOAD_CONST': (0, 1, None),
    'LOAD_DEREF': (0, 1, None),
    'LOAD_FAST': (0, 1, None),
    'LOAD_GLOBAL': (0, 1, None),
    'LOAD_LOCALS': (None, None, None),
    'LOAD_NAME': (0, 1, None),
    'MAKE_CLOSURE': (None, None, None),
    'MAKE_FUNCTION': (-2, 1, None),
    'MAP_ADD': (None, None, None),
    'NOP': (0, None, None),
    'POP_BLOCK': (0, None, 1),
    'POP_EXCEPT': (None, None, None),
    'POP_JUMP_IF_FALSE': (1, None, 1),
    'POP_JUMP_IF_TRUE': (1, None, 1),
    'POP_TOP': (1, None, 1),
    'PRINT_EXPR': (1, None, 1),
    'PRINT_ITEM': (1, None, 1),
    'PRINT_ITEM_TO': (2, None, 1),
    'PRINT_NEWLINE': (0, None, 1),
    'PRINT_NEWLINE_TO': (1, None, 1),
    'RAISE_VARARGS': (None, None, None),
    'RETURN_VALUE': (1, None, 1),
    'ROT_FOUR': (None, None, None),
    'ROT_THREE': (None, None, None),
    'ROT_TWO': (None, None, None),
    'SETUP_EXCEPT': (None, None, None),
    'SETUP_FINALLY': (None, None, None),
    'SETUP_LOOP': (None, None, None),
    'SETUP_WITH': (None, None, None),
    'SET_ADD': (None, None, None),
    'SLICE+0': (1, 1, None),
    'SLICE+1': (2, 1, None),
    'SLICE+2': (2, 1, None),
    'SLICE+3': (3, 1, None),
    'STOP_CODE': (None, None, None),
    'STORE_ATTR': (2, None, 1),
    'STORE_DEREF': (1, 0, 1),
    'STORE_FAST': (1, None, 1),
    'STORE_GLOBAL': (1, None, 1),
    'STORE_LOCALS': (None, None, None),
    'STORE_MAP': (1, None, 1),
    'STORE_NAME': (1, None, 1),
    'STORE_SLICE+0': (1, None, 1),
    'STORE_SLICE+1': (2, None, 1),
    'STORE_SLICE+2': (2, None, 1),
    'STORE_SLICE+3': (3, None, 1),
    'STORE_SUBSCR': (3, None, 1),
    'UNARY_CONVERT': (1, 1, None),
    'UNARY_INVERT': (1, 1, None),
    'UNARY_NEGATIVE': (1, 1, None),
    'UNARY_NOT': (1, 1, None),
    'UNARY_POSITIVE': (1, 1, None),
    'UNPACK_EX': (None, None, None),
    'UNPACK_SEQUENCE': (None, None, None),
    'WITH_CLEANUP': (None, None, None),
    'YIELD_VALUE': (1, None, 1),
}

# ______________________________________________________________________
# Module functions

def itercode(code):
    """Return a generator of byte-offset, opcode, and argument
    from a byte-code-string
    """
    i = 0
    extended_arg = 0
    if isinstance(code[0], str):
        code = [ord(c) for c in code]
    n = len(code)
    while i < n:
        op = code[i]
        num = i
        i = i + 1
        oparg = None
        if op >= opcode.HAVE_ARGUMENT:
            oparg = code[i] + (code[i + 1] * 256) + extended_arg
            extended_arg = 0
            i = i + 2
            if op == opcode.EXTENDED_ARG:
                extended_arg = oparg * 65536

        delta = yield num, op, oparg
        if delta is not None:
            abs_rel, dst = delta
            assert abs_rel == 'abs' or abs_rel == 'rel'
            i = dst if abs_rel == 'abs' else i + dst

# ______________________________________________________________________

def extendlabels(code, labels = None):
    """Extend the set of jump target labels to account for the
    passthrough targets of conditional branches.

    This allows us to create a control flow graph where there is at
    most one branch per basic block.
    """
    if labels is None:
        labels = []
    if isinstance(code[0], str):
        code = [ord(c) for c in code]
    n = len(code)
    i = 0
    while i < n:
        op = code[i]
        i += 1
        if op >= dis.HAVE_ARGUMENT:
            i += 2
            label = -1
            if op in hasjump:
                label = i
            if label >= 0:
                if label not in labels:
                    labels.append(label)
        elif op == opcode.opmap['BREAK_LOOP']:
            if i not in labels:
                labels.append(i)
    return labels

# ______________________________________________________________________

def get_code_object (func):
    return getattr(func, '__code__', getattr(func, 'func_code', None))

# ______________________________________________________________________
# End of opcode_util.py
