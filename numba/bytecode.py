"""
From NumbaPro

"""
from __future__ import print_function, division, absolute_import

import dis
import sys
import inspect
from collections import namedtuple

from numba import utils
from numba.config import PYVERSION

opcode_info = namedtuple('opcode_info', ['argsize'])


def get_function_object(obj):
    """
    Objects that wraps function should provide a "__numba__" magic attribute
    that contains a name of an attribute that contains the actual python
    function object.
    """
    attr = getattr(obj, "__numba__", None)
    if attr:
        return getattr(obj, attr)
    return obj


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
            ('BUILD_SET', 2),
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
            ('STORE_SLICE+0', 0),
            ('STORE_SLICE+1', 0),
            ('STORE_SLICE+2', 0),
            ('STORE_SLICE+3', 0),
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
                    ('BUILD_MAP', 2),
                    ('BUILD_SLICE', 2),
                    ('BUILD_TUPLE', 2),
                    ('CALL_FUNCTION', 2),
                    ('CALL_FUNCTION_VAR', 2),
                    ('COMPARE_OP', 2),
                    ('DELETE_ATTR', 2),
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
                    ('LOAD_DEREF', 2),
                    ('POP_BLOCK', 0),
                    ('POP_TOP', 0),
                    ('RAISE_VARARGS', 2),
                    ('RETURN_VALUE', 0),
                    ('ROT_THREE', 0),
                    ('ROT_TWO', 0),
                    ('SETUP_LOOP', 2),
                    ('STORE_ATTR', 2),
                    ('STORE_FAST', 2),
                    ('STORE_MAP', 0),
                    ('STORE_SUBSCR', 0),
                    ('UNARY_POSITIVE', 0),
                    ('UNARY_NEGATIVE', 0),
                    ('UNARY_INVERT', 0),
                    ('UNARY_NOT', 0),
                    ('UNPACK_SEQUENCE', 2),
                    ('YIELD_VALUE', 0),
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

    @classmethod
    def get(cls, offset, opname, arg):
        return cls(offset, dis.opmap[opname], arg)

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

    @property
    def block_effect(self):
        """Effect of the block stack
        Returns +1 (push), 0 (none) or -1 (pop)
        """
        if self.opname.startswith('SETUP_'):
            return 1
        elif self.opname == 'POP_BLOCK':
            return -1
        else:
            return 0


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
            ts = "offset=%d opcode=0x%x opname=%s"
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
            _offset, byte = next(self.iter)
            buf |= byte << (8 * i)
        return buf


class ByteCodeOperation(object):
    def __init__(self, inst, args):
        self.inst = inst
        self.args = args


class ByteCodeSupportError(Exception):
    pass


class ByteCodeBase(object):
    __slots__ = (
        'func', 'func_name', 'func_qualname', 'filename',
        'pysig', 'co_names', 'co_varnames', 'co_consts', 'co_freevars',
        'table', 'labels', 'arg_count', 'arg_names',
        )

    def __init__(self, func, func_qualname, pysig, filename, co_names,
                 co_varnames, co_consts, co_freevars, table, labels,
                 is_generator, arg_count=None, arg_names=None):
        # When given, these values may not match the pysig's
        # (when lifting loops)
        if arg_count is None:
            arg_count = len(pysig.parameters)
        if arg_names is None:
            arg_names = list(pysig.parameters)
        self.func = func
        self.module = inspect.getmodule(func)
        self.is_generator = is_generator
        self.func_qualname = func_qualname
        self.func_name = func_qualname.split('.')[-1]
        self.pysig = pysig
        self.arg_count = arg_count
        self.arg_names = arg_names
        self.filename = filename
        self.co_names = co_names
        self.co_varnames = co_varnames
        self.co_consts = co_consts
        self.co_freevars = co_freevars
        self.table = table
        self.labels = labels
        self.firstlineno = min(inst.lineno for inst in self.table.values())

    def __iter__(self):
        return utils.itervalues(self.table)

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

        return '\n'.join('%s %10s\t%s' % ((label_marker(i),) + i)
                         for i in utils.iteritems(self.table))


class CustomByteCode(ByteCodeBase):
    """
    A simplified ByteCode class, used for hosting inner loops
    when loop-lifting.
    """


class ByteCode(ByteCodeBase):
    def __init__(self, func):
        func = get_function_object(func)
        code = get_code_object(func)
        pysig = utils.pysignature(func)
        if not code:
            raise ByteCodeSupportError("%s does not provide its bytecode" %
                                       func)
        if code.co_cellvars:
            raise ByteCodeSupportError("does not support cellvars")

        table = utils.SortedMap(ByteCodeIter(code))
        labels = set(dis.findlabels(code.co_code))
        labels.add(0)

        try:
            func_qualname = func.__qualname__
        except AttributeError:
            func_qualname = func.__name__

        self._mark_lineno(table, code)
        super(ByteCode, self).__init__(func=func,
                                       func_qualname=func_qualname,
                                       is_generator=inspect.isgeneratorfunction(func),
                                       pysig=pysig,
                                       filename=code.co_filename,
                                       co_names=code.co_names,
                                       co_varnames=code.co_varnames,
                                       co_consts=code.co_consts,
                                       co_freevars=code.co_freevars,
                                       table=table,
                                       labels=list(sorted(labels)))

    @classmethod
    def _mark_lineno(cls, table, code):
        '''Fill the lineno info for all bytecode inst
        '''
        for offset, lineno in dis.findlinestarts(code):
            if offset in table:
                table[offset].lineno = lineno
        known = -1
        for inst in table.values():
            if inst.lineno >= 0:
                known = inst.lineno
            else:
                inst.lineno = known

