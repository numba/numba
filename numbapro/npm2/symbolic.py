import inspect
import dis
import operator
from contextlib import contextmanager
from collections import defaultdict, namedtuple

from .errors import error_context
from .bytecode import ByteCode

COMPARE_OP_FUNC = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
}

class SymbolicExecution(object):
    def __init__(self, func):
        self.func = func
        self.bytecode = ByteCode(func)
        self.runtime = SymbolicRuntime(self, self.bytecode)

        # ignore any POP_TOP at the first of a block for python26
        self._ignore_first_pop_top = set()

    def strip_dead_block(self):
        dead = []
        # scan
        for blk in self.runtime.blocks:
            if blk.is_dead():
                dead.append(blk)
        # remove
        for blk in dead:
            self.runtime.blocks.remove(blk)

    def visit(self):
        # prepare arguments
        argspec = inspect.getargspec(self.func)
        assert not argspec.defaults, "does not support defaults"
        assert not argspec.varargs, "does not support varargs"
        assert not argspec.keywords, "does not support keywords"
        firstinst = self.bytecode[0]
        for argnum, argname in enumerate(argspec.args):
            self.runtime.load_argument(num=argnum, name=argname,
                                       lineno=firstinst.lineno - 1)
        # visit every bytecode
        for inst in self.bytecode:
            with error_context(inst.lineno):
                if self.runtime.curblock.terminator is not None:
                    self.runtime.curblock = self.runtime.blocks[inst.offset]

                oldblock = self.runtime.curblock
                self.runtime.on_next_inst(inst)         # update curblock


                if oldblock is not self.runtime.curblock:
                    if (inst.opname == 'POP_TOP' and
                           self.runtime.curblock in self._ignore_first_pop_top):
                        continue    # skip
                attr = 'visit_' + inst.opname
                func = getattr(self, attr, self.generic_visit)
                func(inst)

        # run passes
        self.strip_dead_block()
        # generate dominator sets for each block
        self.doms = find_dominators(self.runtime.blocks)

    def generic_visit(self, inst):
        raise NotImplementedError(inst)

    def visit_POP_TOP(self, inst):
        self.runtime.pop()

    def visit_LOAD_FAST(self, inst):
        name = self.runtime.varnames[inst.arg]
        self.runtime.load_name(name, inst.lineno)

    def visit_LOAD_GLOBAL(self, inst):
        name = self.runtime.names[inst.arg]
        self.runtime.load_global(name, inst.lineno)

    def visit_LOAD_CONST(self, inst):
        const = self.runtime.consts[inst.arg]
        self.runtime.load_const(const, inst.lineno)

    def visit_STORE_FAST(self, inst):
        name = self.runtime.varnames[inst.arg]
        self.runtime.store_name(name, self.runtime.pop(), inst.lineno)

    def visit_UNARY_POSITIVE(self, inst):
        self.runtime.unary_op(operator.pos, inst.lineno)

    def visit_UNARY_NEGATIVE(self, inst):
        self.runtime.unary_op(operator.neg, inst.lineno)

    def visit_UNARY_INVERT(self, inst):
        self.runtime.unary_op(operator.invert, inst.lineno)

    def visit_UNARY_NOT(self, inst):
        self.runtime.unary_op(operator.not_, inst.lineno)

    def visit_BINARY_ADD(self, inst):
        self.runtime.binary_op(operator.add, inst.lineno)

    def visit_BINARY_SUBTRACT(self, inst):
        self.runtime.binary_op(operator.sub, inst.lineno)

    def visit_BINARY_MULTIPLY(self, inst):
        self.runtime.binary_op(operator.mul, inst.lineno)

    def visit_BINARY_DIVIDE(self, inst):
        self.runtime.binary_op(operator.floordiv, inst.lineno)

    def visit_BINARY_FLOOR_DIVIDE(self, inst):
        self.runtime.binary_op(operator.floordiv, inst.lineno)

    def visit_BINARY_TRUE_DIVIDE(self, inst):
        self.runtime.binary_op(operator.truediv, inst.lineno)

    def visit_BINARY_MODULO(self, inst):
        self.runtime.binary_op(operator.mod, inst.lineno)

    def visit_BINARY_POWER(self, inst):
        self.runtime.binary_op(operator.pow, inst.lineno)

    def visit_BINARY_RSHIFT(self, inst):
        self.runtime.binary_op(operator.rshift, inst.lineno)

    def visit_BINARY_LSHIFT(self, inst):
        self.runtime.binary_op(operator.lshift, inst.lineno)

    def visit_BINARY_AND(self, inst):
        self.runtime.binary_op(operator.and_, inst.lineno)

    def visit_BINARY_OR(self, inst):
        self.runtime.binary_op(operator.or_, inst.lineno)

    def visit_BINARY_XOR(self, inst):
        self.runtime.binary_op(operator.xor, inst.lineno)

    def visit_INPLACE_ADD(self, inst):
        self.runtime.binary_op(operator.add, inst.lineno)

    def visit_INPLACE_SUBTRACT(self, inst):
        self.runtime.binary_op(operator.sub, inst.lineno)

    def visit_INPLACE_MULTIPLY(self, inst):
        self.runtime.binary_op(operator.mul, inst.lineno)

    def visit_INPLACE_DIVIDE(self, inst):
        self.runtime.binary_op(operator.floordiv, inst.lineno)

    def visit_INPLACE_FLOOR_DIVIDE(self, inst):
        self.runtime.binary_op(operator.floordiv, inst.lineno)

    def visit_INPLACE_TRUE_DIVIDE(self, inst):
        self.runtime.binary_op(operator.truediv, inst.lineno)

    def visit_INPLACE_MODULO(self, inst):
        self.runtime.binary_op(operator.mod, inst.lineno)

    def visit_INPLACE_POWER(self, inst):
        self.runtime.binary_op(operator.pow, inst.lineno)

    def visit_INPLACE_RSHIFT(self, inst):
        self.runtime.binary_op(operator.rshift, inst.lineno)

    def visit_INPLACE_LSHIFT(self, inst):
        self.runtime.binary_op(operator.lshift, inst.lineno)

    def visit_INPLACE_AND(self, inst):
        self.runtime.binary_op(operator.and_, inst.lineno)

    def visit_INPLACE_OR(self, inst):
        self.runtime.binary_op(operator.or_, inst.lineno)

    def visit_INPLACE_XOR(self, inst):
        self.runtime.binary_op(operator.xor, inst.lineno)

    def visit_COMPARE_OP(self, inst):
        opfunc = COMPARE_OP_FUNC[dis.cmp_op[inst.arg]]
        self.runtime.binary_op(opfunc, inst.lineno)

    def visit_SETUP_LOOP(self, inst):
        self.runtime.setup_loop(inst.arg + inst.next)

    def visit_BREAK_LOOP(self, inst):
        self.runtime.break_loop(inst.next, inst.lineno)

    def visit_CALL_FUNCTION(self, inst):
        argc = inst.arg & 0xff
        kwsc = (inst.arg >> 8) & 0xff
        self.runtime.setup_call(argc, kwsc, inst.lineno)

    def visit_GET_ITER(self, inst):
        self.runtime.get_iter(inst.lineno)

    def visit_FOR_ITER(self, inst):
        self.runtime.for_iter(inst.arg + inst.next, inst.offset, inst.next,
                              inst.lineno)

    def visit_JUMP_IF_TRUE_OR_POP(self, inst):
        raise Exception('do not support `a = b and c or d`')

    def visit_JUMP_IF_FALSE_OR_POP(self, inst):
        raise Exception('do not support `a = b and c or d`')

    def visit_POP_JUMP_IF_TRUE(self, inst):
        falsebr = self.runtime.blocks[inst.next]
        truebr = self.runtime.blocks[inst.arg]
        self.runtime.jump_if(truebr, falsebr, inst.lineno)

    def visit_POP_JUMP_IF_FALSE(self, inst):
        truebr = self.runtime.blocks[inst.next]
        falsebr = self.runtime.blocks[inst.arg]
        self.runtime.jump_if(truebr, falsebr, inst.lineno)

    def visit_POP_BLOCK(self, inst):
        self.runtime.pop_block(inst.lineno)

    def visit_JUMP_IF_TRUE(self, inst):
        falsebr = self.runtime.blocks[inst.next]
        truebr = self.runtime.blocks[inst.arg]
        self._ignore_first_pop_top.add(truebr)
        self._ignore_first_pop_top.add(falsebr)
        self.runtime.jump_if(truebr, falsebr, inst.lineno)

    def visit_JUMP_IF_FALSE(self, inst):
        truebr = self.runtime.blocks[inst.next]
        falsebr = self.runtime.blocks[inst.arg]
        self._ignore_first_pop_top.add(truebr)
        self._ignore_first_pop_top.add(falsebr)
        self.runtime.jump_if(truebr, falsebr, inst.lineno)

    def visit_JUMP_ABSOLUTE(self, inst):
        target = self.runtime.blocks[inst.arg]
        self.runtime.jump(target, inst.lineno)

    def visit_JUMP_FORWARD(self, inst):
        target = self.runtime.blocks[inst.arg + inst.next]
        self.runtime.jump(target, inst.lineno)

    def visit_RETURN_VALUE(self, inst):
        self.runtime.ret(inst.lineno)

#---------------------------------------------------------------------------
# Passes

def find_dominators(blocks):
    doms = {}
    for b in blocks:
        doms[b] = set()

    doms[blocks[0]].add(blocks[0])
    allblks = set(blocks)

    remainblks = frozenset(blk for blk in blocks if blk.offset != 0)
    for blk in remainblks:
        doms[blk] |= allblks

    changed = True
    while changed:
        changed = False
        for blk in remainblks:
            d = doms[blk]
            ps = [doms[p] for p in blk.incoming_blocks]
            p = reduce(set.intersection, ps)
            new = set([blk]) | p
            if new != d:
                doms[blk] = new
                changed = True

    return doms

#---------------------------------------------------------------------------
# Internals

setup_loop_info = namedtuple('setup_loop_info', ['sp', 'block', 'nextblock'])

class SymbolicRuntime(object):
    def __init__(self, parent, bytecode):
        self.parent = parent

        self.varnames = bytecode.code.co_varnames
        self.consts = bytecode.code.co_consts
        self.names = bytecode.code.co_names
        self.blocks = BlockMap()
        self.curblock = self.blocks[0]
        self.scopes = []

    @property
    def stack(self):
        return self.curblock.stack

    def on_next_inst(self, inst):
        oldblock = self.curblock
        self.curblock = self.blocks.get(inst.offset, oldblock)

    def setup_loop(self, offset):
        self.scopes.append(setup_loop_info(sp=len(self.stack),
                                           block=self.curblock,
                                           nextblock=self.blocks[offset]))

    def break_loop(self, next, lineno):
        target = self.scopes[-1].nextblock
        self.terminate(Inst('jump', lineno, target=target))
        self.curblock.connect(target)
        self.curblock = self.blocks[next]
    
    def pop_block(self, lineno):
        del self.stack[self.scopes[-1].sp:]
        self.scopes.pop()

    def setup_call(self, argc, kwsc, lineno):
        def pop_kws():
            val = self.pop()
            key = self.pop()
            if key.opcode != 'const':
                raise ArgumentError('keyword must be a constant')
            return key.value, val

        kws = list(reversed([pop_kws() for i in range(kwsc)]))
        args = list(reversed([self.pop() for i in range(argc)]))

        func = self.pop()
        self.call(lineno, func, args=args, kws=kws)


    def push(self, value):
        "push value onto the stack"
        self.stack.append(value)

    def pop(self):
        "pop value from the stack"
        return self.stack.pop()

    def peek(self):
        "peek at the TOS"
        return self.stack[-1]

    def insert(self, inst):
        inst.block = self.curblock
        self.curblock.code.append(inst)
        return inst

    def terminate(self, inst):
        assert self.curblock.terminator is None
        inst.block = self.curblock
        self.curblock.terminator = inst

    def store_name(self, name, value, lineno):
        self.insert(Inst('store', lineno, name=name, value=value))

    def load_argument(self, num, name, lineno):
        inst = Inst('arg', lineno, argnum=num)
        ref = self.insert(inst)
        self.store_name(name, inst, lineno)

    def load_name(self, name, lineno):
        self.push(self.insert(Inst('load', lineno, name=name)))

    def load_global(self, name, lineno):
        self.push(self.insert(Inst('global', lineno, name=name)))

    def load_const(self, value, lineno):
        self.push(self.insert(Inst('const', lineno, value=value)))

    def get_iter(self, lineno):
        obj = self.pop()
        self.push(self.insert(Inst('iter', lineno, obj=obj)))


    def for_iter(self, delta, ipcur, ipnext, lineno):
        iterobj = self.peek()

        loophead = self.blocks[ipcur]
        self.terminate(Inst('jump', lineno, target=loophead))
        self.curblock.connect(loophead)
        self.curblock = loophead

        predloop = self.insert(Inst('for_valid', lineno, iter=iterobj))
        loopbody = self.blocks[ipnext]
        loopexit = self.blocks[delta]
        self.terminate(Inst('branch', lineno, cond=predloop,
                            truebr=loopbody, falsebr=loopexit))
        self.curblock.connect(loopbody)
        self.curblock.connect(loopexit)
        self.curblock = loopbody
        
        self.push(self.insert(Inst('for_next', lineno, iter=iterobj)))

    def call(self, lineno, func, args=(), kws=()):
        callinst = Inst('call', lineno, callee=func, args=args, kws=kws)
        self.push(self.insert(callinst))

    def unary_op(self, op, lineno):
        tos = self.pop()
        self.call(lineno, op, args=(tos,))

    def binary_op(self, op, lineno):
        rhs = self.pop()
        lhs = self.pop()
        self.call(lineno, op, args=(lhs, rhs))

    def jump_if(self, truebr, falsebr, lineno):
        cond = self.pop()
        self.terminate(Inst('branch', lineno, cond=cond, truebr=truebr, falsebr=falsebr))
        self.curblock.connect(truebr)
        self.curblock.connect(falsebr)

    def jump(self, target, lineno):
        self.terminate(Inst('jump', lineno, target=target))
        self.curblock.connect(target)

    def ret(self, lineno):
        val = self.pop()
        if val.opcode == 'const' and val.value is None:
            self.terminate(Inst('retvoid', lineno))
        else:
            self.terminate(Inst('ret', lineno, value=val))


class BlockMap(object):
    def __init__(self):
        self._map = {}

    def __getitem__(self, offset):
        try:
            return self._map[offset]
        except KeyError:
            self._map[offset] = Block(offset)
            return self._map[offset]

    def __setitem__(self, offset):
        del self._sorted
        return self._map[offset]

    def get(self, key, default):
        if key not in self._map:
            return default
        else:
            return self._map[key]

    def remove(self, blk):
        del self._sorted
        del self._map[blk.offset]

    def sorted(self):
        try:
            return self._sorted
        except AttributeError:
            self._sorted = sorted(self._map.iteritems())
            return self._sorted

    def __iter__(self):
        return iter(v for k, v in self.sorted())

class Incomings(object):
    def __init__(self):
        self.incomings = {}

    def __repr__(self):
        ins = '; '.join('%r=%r' % it for it in self.incomings.iteritems())
        return ins

    def __setitem__(self, k, v):
        self.incomings[k] = v

class Block(object):
    def __init__(self, offset):
        self.offset = offset
        self.code = []
        self.stack = []
        self.terminator = None
        self.incoming_blocks = set()
        self.outgoing_blocks = set()

    def is_empty(self):
        return not self.code

    def is_dead(self):
        return not self.incoming_blocks and not self.outgoing_blocks

    def connect(self, nextblk):
        nextblk.incoming_blocks.add(self)
        self.outgoing_blocks.add(nextblk)

    def descr(self):
        ins = ', '.join(str(b.offset) for b in self.incoming_blocks)
        head = ["block %4d        ; incoming %s" % (self.offset, ins)]
        body = ["    {!r:<30} = {!s}".format(c, c) for c in self.code]
        tail = ["    %s" % self.terminator]
        buf = head + body + tail
        return '\n'.join(buf)

    def __str__(self):
        return self.descr()

    def __repr__(self):
        return '<block %d>' % self.offset

class Inst(object):
    def __init__(self, opcode, lineno, **kwargs):
        self.opcode = opcode
        self.lineno = lineno
        self.attrs = set(kwargs.keys())
        self.block = None
        for k, v in kwargs.items():
            assert not hasattr(self, k)
            setattr(self, k, v)

    def list_attrs(self):
        return ((k, getattr(self, k)) for k in self.attrs)

    def __str__(self):
        attrs = ', '.join('%s=%1r' % (k, v) for k, v in self.list_attrs())
        return '%s(%s)' % (self.opcode, attrs)

    def __repr__(self):
        return '<%s 0x%x>' % (self.opcode, id(self))

    def update(self, **kws):
        for k, v in kws.iteritems():
            if hasattr(self, k):
                assert k in self.attrs
            setattr(self, k, v)
            self.attrs.add(k)

    def __contains__(self, attrname):
        return attrname not in self.attrs

