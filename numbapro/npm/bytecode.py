import dis, sys
from collections import namedtuple

from .utils import SortedMap

opcode_info = namedtuple('opcode_info', ['argsize'])

def get_code_object(obj):
    "Shamelessly borrowed from llpython"
    return getattr(obj, '__code__', getattr(obj, 'func_code', None))

def _make_bytecode_table(seq):
    return dict((dis.opmap[opname], opcode_info(argsize=argsize))
                for opname, argsize in seq)

if sys.version_info[:2] == (2, 6):  # python 2.6
    BYTECODE_VERSION_SPECIFIC = [
        ('JUMP_IF_FALSE', 2),
        ('JUMP_IF_TRUE', 2),
    ]
elif sys.version_info[:2] >= (2, 7):  # python 2.7
    BYTECODE_VERSION_SPECIFIC = [
        ('POP_JUMP_IF_FALSE', 2),
        ('POP_JUMP_IF_TRUE', 2),
        ('JUMP_IF_TRUE_OR_POP', 2),
        ('JUMP_IF_FALSE_OR_POP', 2),
    ]

BYTECODES = [
    # opname, operandlen
    ('BINARY_ADD', 0),
    ('BINARY_DIVIDE', 0),
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
    ('UNARY_POSITIVE', 0),
    ('UNARY_NEGATIVE', 0),
    ('UNARY_INVERT', 0),
    ('UNARY_NOT', 0),
    ('BUILD_TUPLE', 2),
    ('CALL_FUNCTION', 2),
    ('COMPARE_OP', 2),
    ('DUP_TOP',  0),
    ('DUP_TOPX', 2),
    ('FOR_ITER', 2),
    ('GET_ITER', 0),
    ('INPLACE_ADD', 0),
    ('INPLACE_SUBTRACT', 0),
    ('INPLACE_MULTIPLY', 0),
    ('INPLACE_DIVIDE', 0),
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
    ('RETURN_VALUE', 0),
    ('ROT_THREE', 0),
    ('ROT_TWO', 0),
    ('SETUP_LOOP', 2),
    ('STORE_FAST', 2),
#    ('STORE_ATTR', 2), # not supported
    ('STORE_SUBSCR', 0),
    ('UNPACK_SEQUENCE', 2),
    ('SLICE+0', 0),
    ('SLICE+1', 0),
    ('SLICE+2', 0),
    ('SLICE+3', 0),
    ('RAISE_VARARGS', 2),
    ('BUILD_SLICE', 2),
] + BYTECODE_VERSION_SPECIFIC


BYTECODE_TABLE = _make_bytecode_table(BYTECODES)


JUMP_OPS = ['JUMP_ABSOLUTE', 'JUMP_FORWARD',
            'POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE',
            'JUMP_IF_FALSE', 'JUMP_IF_TRUE',
            'JUMP_IF_TRUE_OR_POP', 'JUMP_IF_FALSE_OR_POP',
            'FOR_ITER', ]


class ByteCodeInst(object):
    '''
    offset: byte offset of opcode
    opcode: opcode integer value
    arg: instruction arg
    lineno: -1 means unknown
    '''
    def __init__(self, offset, opcode, arg):
        self.offset = offset
        self.next = offset + BYTECODE_TABLE[opcode].argsize + 1
        self.opcode = opcode
        self.opname = dis.opname[opcode]
        self.arg = arg
        self.lineno = -1  # unknown line number

    def is_jump(self):
        return self.opname in JUMP_OPS

    def __repr__(self):
        return '%s(arg=%s, lineno=%d)' % (self.opname, self.arg, self.lineno)

class ByteCodeIter(object):
    def __init__(self, code):
        self.code = code
        self.iter = ((i, ord(x)) for i, x in enumerate(self.code.co_code))

    def __iter__(self):
        return self

    def next(self):
        offset, opcode = self.iter.next()
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

    def read_arg(self, size):
        buf = 0
        for i in range(size):
            _offset, byte = self.iter.next()
            buf |= byte << (8 * i)
        return buf

class ByteCodeOperation(object):
    def __init__(self, inst, args):
        self.inst = inst
        self.args = args

class ByteCode(object):
    def __init__(self, func):
        self.code = get_code_object(func)
        if not self.code:
            raise Exception("%s does not provide its bytecode" % func)
        #print dis.dis(self.code)
        assert not self.code.co_freevars, "does not support freevars"
        assert not self.code.co_cellvars, "does not support cellvars"
        self.table = SortedMap(ByteCodeIter(self.code))

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
        return self.table.itervalues()

    def __getitem__(self, offset):
        return self.table[offset]

    def __contains__(self, offset):
        return offset in self.table

    def dump(self):
        return '\n'.join('%10d\t%s' % i for i in self.table.iteritems())

