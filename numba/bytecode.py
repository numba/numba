"""
From NumbaPro

"""
from __future__ import print_function, division, absolute_import
import dis
import sys
from collections import namedtuple
from numba import utils
from numba.config import PYVERSION

opcode_info = namedtuple('opcode_info', ['argsize'])


def get_code_object(obj):
    "Shamelessly borrowed from llpython"
    return getattr(obj, '__code__', getattr(obj, 'func_code', None))


def _make_bytecode_table():
    if sys.version_info[:2] == (2, 6):  # python 2.6
        version_specific = [
            ('JUMP_IF_FALSE', 2),
            ('JUMP_IF_TRUE', 2),
        ]

    elif sys.version_info[:2] >= (2, 7):  # python 2.7+
        version_specific = [
            ('POP_JUMP_IF_FALSE', 2),
            ('POP_JUMP_IF_TRUE', 2),
            ('JUMP_IF_TRUE_OR_POP', 2),
            ('JUMP_IF_FALSE_OR_POP', 2),
        ]

    if sys.version_info[0] == 2:
        version_specific += [
            ('BINARY_DIVIDE', 0),
            ('DUP_TOPX', 2),
            ('INPLACE_DIVIDE', 0),
            ('PRINT_ITEM', 0),
            ('PRINT_NEWLINE', 0),
            ('SLICE+0', 0),
            ('SLICE+1', 0),
            ('SLICE+2', 0),
            ('SLICE+3', 0),
        ]
    elif sys.version_info[0] == 3:
        version_specific += [
            ('DUP_TOP_TWO', 0)
        ]

    bytecodes = [
                    # opname, operandlen
                    ('BINARY_ADD', 0),
                    ('BINARY_TRUE_DIVIDE', 0),
                    ('BINARY_MULTIPLY', 0),
                    ('BINARY_SUBSCR', 0),
                    ('BINARY_SUBTRACT', 0),
                    ('BINARY_FLOOR_DIVIDE', 0),
                    ('BINARY_MODULO', 0),
                    ('BINARY_POWER', 0),
                    ('BINARY_AND', 0),
                    ('BINARY_OR', 0),
                    ('BINARY_XOR', 0),
                    ('BINARY_LSHIFT', 0),
                    ('BINARY_RSHIFT', 0),
                    ('BREAK_LOOP', 0),
                    ('BUILD_LIST', 2),
                    ('BUILD_SLICE', 2),
                    ('BUILD_TUPLE', 2),
                    ('CALL_FUNCTION', 2),
                    ('COMPARE_OP', 2),
                    ('DUP_TOP', 0),
                    ('FOR_ITER', 2),
                    ('GET_ITER', 0),
                    ('INPLACE_ADD', 0),
                    ('INPLACE_SUBTRACT', 0),
                    ('INPLACE_MULTIPLY', 0),
                    ('INPLACE_TRUE_DIVIDE', 0),
                    ('INPLACE_FLOOR_DIVIDE', 0),
                    ('INPLACE_MODULO', 0),
                    ('INPLACE_POWER', 0),
                    ('INPLACE_AND', 0),
                    ('INPLACE_OR', 0),
                    ('INPLACE_XOR', 0),
                    ('INPLACE_LSHIFT', 0),
                    ('INPLACE_RSHIFT', 0),
                    ('JUMP_ABSOLUTE', 2),
                    ('JUMP_FORWARD', 2),
                    ('LOAD_ATTR', 2),
                    ('LOAD_CONST', 2),
                    ('LOAD_FAST', 2),
                    ('LOAD_GLOBAL', 2),
                    ('POP_BLOCK', 0),
                    ('POP_TOP', 0),
                    ('RAISE_VARARGS', 2),
                    ('RETURN_VALUE', 0),
                    ('ROT_THREE', 0),
                    ('ROT_TWO', 0),
                    ('SETUP_LOOP', 2),
                    ('STORE_FAST', 2),
                    #    ('STORE_ATTR', 2), # not supported
                    ('STORE_SUBSCR', 0),
                    ('UNARY_POSITIVE', 0),
                    ('UNARY_NEGATIVE', 0),
                    ('UNARY_INVERT', 0),
                    ('UNARY_NOT', 0),
                    ('UNPACK_SEQUENCE', 2),
                ] + version_specific

    return dict((dis.opmap[opname], opcode_info(argsize=argsize))
                for opname, argsize in bytecodes)


def _as_opcodes(seq):
    lst = []
    for s in seq:
        c = dis.opmap.get(s)
        if c is not None:
            lst.append(c)
    return lst

BYTECODE_TABLE = _make_bytecode_table()

JREL_OPS = frozenset(dis.hasjrel)
JABS_OPS = frozenset(dis.hasjabs)
JUMP_OPS = JREL_OPS | JABS_OPS
TERM_OPS = frozenset(_as_opcodes(['RETURN_VALUE', 'RAISE_VARARGS']))


class ByteCodeInst(object):
    '''
    Attributes
    ----------
    - offset:
        byte offset of opcode
    - opcode:
        opcode integer value
    - arg:
        instruction arg
    - lineno:
        -1 means unknown
    '''
    __slots__ = 'offset', 'next', 'opcode', 'opname', 'arg', 'lineno'

    def __init__(self, offset, opcode, arg):
        self.offset = offset
        self.next = offset + BYTECODE_TABLE[opcode].argsize + 1
        self.opcode = opcode
        self.opname = dis.opname[opcode]
        self.arg = arg
        self.lineno = -1  # unknown line number

    @property
    def is_jump(self):
        return self.opcode in JUMP_OPS

    @property
    def is_terminator(self):
        return self.opcode in TERM_OPS

    def get_jump_target(self):
        assert self.is_jump
        if self.opcode in JREL_OPS:
            return self.next + self.arg
        else:
            assert self.opcode in JABS_OPS
            return self.arg

    def __repr__(self):
        return '%s(arg=%s, lineno=%d)' % (self.opname, self.arg, self.lineno)


class ByteCodeIter(object):
    def __init__(self, code):
        self.code = code
        if PYVERSION > (3, 0):
            self.iter = enumerate(self.code.co_code)
        else:
            self.iter = ((i, ord(x)) for i, x in enumerate(self.code.co_code))

    def __iter__(self):
        return self

    def next(self):
        offset, opcode = next(self.iter)
        try:
            info = BYTECODE_TABLE[opcode]
        except KeyError:
            ts = "offset=%d opcode=%x opname=%s"
            tv = offset, opcode, dis.opname[opcode]
            raise NotImplementedError(ts % tv)
        if info.argsize:
            arg = self.read_arg(info.argsize)
        else:
            arg = None
        return offset, ByteCodeInst(offset=offset, opcode=opcode, arg=arg)

    __next__ = next

    def read_arg(self, size):
        buf = 0
        for i in range(size):
            _offset, byte = utils.iter_next(self.iter)
            buf |= byte << (8 * i)
        return buf


class ByteCodeOperation(object):
    def __init__(self, inst, args):
        self.inst = inst
        self.args = args


class ByteCodeSupportError(Exception):
    pass


class ByteCode(object):
    def __init__(self, func):
        self.func = func
        self.code = get_code_object(func)
        self.filename = self.code.co_filename

        # Do basic checking on support for the given bytecode
        if not self.code:
            raise ByteCodeSupportError("%s does not provide its bytecode" %
                                       func)
        if self.code.co_freevars:
            raise ByteCodeSupportError("does not support freevars")
        if self.code.co_cellvars:
            raise ByteCodeSupportError("does not support cellvars")

        self.table = utils.SortedMap(ByteCodeIter(self.code))

        labels = set(dis.findlabels(self.code.co_code))
        labels.add(0)
        self.labels = list(sorted(labels))
        self._mark_lineno()

    def _mark_lineno(self):
        '''Fill the lineno info for all bytecode inst
        '''
        for offset, lineno in dis.findlinestarts(self.code):
            if offset in self.table:
                self.table[offset].lineno = lineno
        known = -1
        for inst in self:
            if inst.lineno >= 0:
                known = inst.lineno
            else:
                inst.lineno = known

    def __iter__(self):
        return utils.dict_itervalues(self.table)

    def __getitem__(self, offset):
        return self.table[offset]

    def __contains__(self, offset):
        return offset in self.table

    def dump(self):
        def label_marker(i):
            if i[1].offset in self.labels:
                return '>'
            else:
                return ' '

        return '\n'.join('%s %10d\t%s' % ((label_marker(i),) + i)
                         for i in utils.dict_iteritems(self.table))

