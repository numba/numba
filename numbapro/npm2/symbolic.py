import inspect
import textwrap
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
        self.blocks = BlockMap()
        for offset in self.bytecode.labels:
            self.blocks[offset]

        self.pending_run = set([0])
        self.processed = set()

        self.scopes = []

        self.varnames = self.bytecode.code.co_varnames
        self.consts = self.bytecode.code.co_consts
        self.names = self.bytecode.code.co_names

    def interpret(self):
        # interpretation loop
        while self.pending_run:
            offset = min(self.pending_run)
            self.pending_run.discard(offset)
            if offset not in self.processed:    # don't repeat the work
                self.processed.add(offset)

                self.curblock = self.blocks[offset]
                if offset == 0:
                    self.push_arguments()

                while offset in self.bytecode:
                    inst = self.bytecode[offset]
                    self.op(inst)
                    offset = inst.next
                    if self.curblock.is_terminated():
                        break

                    if offset in self.blocks:
                        self.pending_run.add(offset)
                        if not self.curblock.is_terminated():
                            self.jump(target=self.blocks[offset])
                        break

        # run passes
        self.doms = find_dominators(self.blocks)
        self.mark_backbone()
        self.strip_dead_block()
        self.complete_phis()
        self.rename_definition()

    def rename_definition(self):
        RenameDefinition(self.blocks, self.doms, self.backbone).run()

    def mark_backbone(self):
        '''The backbone control flow path.
        Variable can only be defined on this path
        '''
        self.backbone = self.doms[self.blocks.last()]

    def complete_phis(self):
        for blk in self.blocks:
            for pos, inst in enumerate(blk.code):
                if inst.opcode == 'phi':
                    for ib in blk.incoming_blocks:
                        try:
                            inst.phi.add_incoming(ib, ib.stack[-1-pos])
                        except IndexError:
                            pass
                else:
                    break

    def strip_dead_block(self):
        dead = []
        # scan
        for blk in self.blocks:
            if blk.offset != 0 and blk.is_dead():
                dead.append(blk)
        # remove
        for blk in dead:
            self.blocks.remove(blk)

    def push_arguments(self):
        argspec = inspect.getargspec(self.func)
        assert not argspec.defaults, "does not support defaults"
        assert not argspec.varargs, "does not support varargs"
        assert not argspec.keywords, "does not support keywords"
        # guess line no
        self.lineno = self.bytecode[0].lineno - 1
        for argnum, argname in enumerate(argspec.args):
            arg = self.insert('arg', num=argnum, name=argname)
            self.push_insert('store', name=argname, value=arg)

    def op(self, inst):
        with error_context(lineno=inst.lineno):
            self.lineno = inst.lineno
            attr = 'op_%s' % inst.opname
            fn = getattr(self, attr, self.generic_op)
            fn(inst)

    def generic_op(self, inst):
        raise NotImplementedError(inst)

    @property
    def stack(self):
        return self.curblock.stack

    def insert(self, op, **kws):
        inst = Inst(op, self.lineno, **kws)
        inst.block = self.curblock
        self.curblock.code.append(inst)
        return inst
        

    def _prepend_phi(self, op, **kws):
        inst = Inst(op, self.lineno, **kws)
        inst.block = self.curblock
        self.curblock.code = [inst] + self.curblock.code
        return inst

    def push_insert(self, op, **kws):
        inst = self.insert(op, **kws)
        self.push(inst)
        return inst

    def push(self, val):
        self.stack.append(val)

    def _insert_phi(self):
        phi = Inst('phi', self.lineno, phi=Incomings())
        phi.block = self.curblock
        pos = 0
        for pos, inst in enumerate(self.curblock.code):
            if inst.opcode != 'phi':
                break
        self.curblock.code.insert(pos, phi)
        return phi

    def peek(self):
        if not self.stack:
            return self._insert_phi()
        else:
            return self.stack[-1]

    def pop(self):
        if not self.stack:
            return self._insert_phi()
        else:
            return self.stack.pop()

    def call(self, func, args=(), kws=()):
        self.push_insert('call', callee=func, args=args, kws=kws)

    def binary_op(self, op):
        rhs = self.pop()
        lhs = self.pop()
        self.call(op, args=(lhs, rhs))

    def unary_op(self, op):
        tos = self.pop()
        self.call(op, args=(tos,))

    def jump(self, target):
        self.pending_run.add(target.offset)
        self.terminate('jump', target=target)
        self.curblock.connect(target)

    def jump_if(self, cond, truebr, falsebr):
        self.pending_run.add(truebr.offset)
        self.pending_run.add(falsebr.offset)
        self.terminate('branch', cond=cond, truebr=truebr, falsebr=falsebr)
        self.curblock.connect(truebr)
        self.curblock.connect(falsebr)

    def terminate(self, op, **kws):
        assert not self.curblock.is_terminated()
        self.curblock.terminator = Inst(op, self.lineno, **kws)

    # ------ op_* ------- #
    
    def op_POP_JUMP_IF_TRUE(self, inst):
        falsebr = self.blocks[inst.next]
        truebr = self.blocks[inst.arg]
        self.jump_if(self.pop(), truebr, falsebr)

    def op_POP_JUMP_IF_FALSE(self, inst):
        truebr = self.blocks[inst.next]
        falsebr = self.blocks[inst.arg]
        self.jump_if(self.pop(), truebr, falsebr)

    def op_JUMP_IF_TRUE(self, inst):
        falsebr = self.blocks[inst.next]
        truebr = self.blocks[inst.arg]
        self.jump_if(self.peek(), truebr, falsebr)

    def op_JUMP_IF_FALSE(self, inst):
        truebr = self.blocks[inst.next]
        falsebr = self.blocks[inst.arg]
        self.jump_if(self.peek(), truebr, falsebr)

    def op_JUMP_IF_TRUE_OR_POP(self, inst):
        falsebr = self.blocks[inst.next]
        truebr = self.blocks[inst.arg]
        self.jump_if(self.peek(), truebr, falsebr)

    def op_JUMP_IF_TRUE_OR_POP(self, inst):
        truebr = self.blocks[inst.next]
        falsebr = self.blocks[inst.arg]
        self.jump_if(self.peek(), truebr, falsebr)

    def op_JUMP_ABSOLUTE(self, inst):
        target = self.blocks[inst.arg]
        self.jump(target)

    def op_JUMP_FORWARD(self, inst):
        target = self.blocks[inst.next + inst.arg]
        self.jump(target)

    def op_RETURN_VALUE(self, inst):
        val = self.pop()
        if val.opcode == 'const' and val.value is None:
            self.terminate('retvoid')
        else:
            self.terminate('ret', value=val)

    def op_SETUP_LOOP(self, inst):
        self.scopes.append((inst.next, inst.next + inst.arg))

    def op_POP_BLOCK(self, inst):
        self.scopes.pop()

    def op_CALL_FUNCTION(self, inst):
        argc = inst.arg & 0xff
        kwsc = (inst.arg >> 8) & 0xff

        def pop_kws():
            val = self.pop()
            key = self.pop()
            if key.opcode != 'const':
                raise ArgumentError('keyword must be a constant')
            return key.value, val

        kws = list(reversed([pop_kws() for i in range(kwsc)]))
        args = list(reversed([self.pop() for i in range(argc)]))

        func = self.pop()
        self.call(func, args=args, kws=kws)

    def op_GET_ITER(self, inst):
        self.call(iter, args=(self.pop(),))

    def op_FOR_ITER(self, inst):
        iterobj = self.peek()
        delta = inst.arg
        loopexit = self.blocks[inst.next + delta]
        loopbody = self.blocks[inst.next]
        self.call('itervalid', args=(iterobj,))
        pred = self.pop()
        self.jump_if(cond=pred, truebr=loopbody, falsebr=loopexit)

        self.curblock, oldblock = loopbody, self.curblock
        self.call('iternext', args=(iterobj,))

        self.curblock = oldblock

    def op_BREAK_LOOP(self, inst):
        scope = self.scopes[-1]
        self.jump(target=self.blocks[scope[1]])

    def op_LOAD_ATTR(self, inst):
        attr = self.names[inst.arg]
        obj = self.pop()
        self.call('.%s' % attr, args=(obj,))

    def op_LOAD_GLOBAL(self, inst):
        name = self.names[inst.arg]
        self.push_insert('global', name=name)

    def op_LOAD_FAST(self, inst):
        name = self.varnames[inst.arg]
        self.push_insert('load', name=name)

    def op_LOAD_CONST(self, inst):
        const = self.consts[inst.arg]
        self.push_insert('const', value=const)

    def op_STORE_FAST(self, inst):
        tos = self.pop()
        name = self.varnames[inst.arg]
        self.insert('store', name=name, value=tos)

    def op_COMPARE_OP(self, inst):
        opfunc = COMPARE_OP_FUNC[dis.cmp_op[inst.arg]]
        self.binary_op(opfunc)

    def op_UNARY_POSITIVE(self, inst):
        self.unary_op(operator.pos)

    def op_UNARY_NEGATIVE(self, inst):
        self.unary_op(operator.neg)

    def op_UNARY_INVERT(self, inst):
        self.unary_op(operator.invert)

    def op_UNARY_NOT(self, inst):
        self.unary_op(operator.not_)

    def op_BINARY_SUBSCR(self, inst):
        self.binary_op(operator.getitem)

    def op_BINARY_ADD(self, inst):
        self.binary_op(operator.add)

    def op_BINARY_SUBTRACT(self, inst):
        self.binary_op(operator.sub)

    def op_BINARY_MULTIPLY(self, inst):
        self.binary_op(operator.mul)

    def op_BINARY_DIVIDE(self, inst):
        self.binary_op(operator.floordiv)

    def op_BINARY_FLOOR_DIVIDE(self, inst):
        self.binary_op(operator.floordiv)

    def op_BINARY_TRUE_DIVIDE(self, inst):
        self.binary_op(operator.truediv)

    def op_BINARY_MODULO(self, inst):
        self.binary_op(operator.mod)

    def op_BINARY_POWER(self, inst):
        self.binary_op(operator.pow)

    def op_BINARY_RSHIFT(self, inst):
        self.binary_op(operator.rshift)

    def op_BINARY_LSHIFT(self, inst):
        self.binary_op(operator.lshift)

    def op_BINARY_AND(self, inst):
        self.binary_op(operator.and_)

    def op_BINARY_OR(self, inst):
        self.binary_op(operator.or_)

    def op_BINARY_XOR(self, inst):
        self.binary_op(operator.xor)

    def op_INPLACE_ADD(self, inst):
        self.binary_op(operator.add)

    def op_INPLACE_SUBTRACT(self, inst):
        self.binary_op(operator.sub)

    def op_INPLACE_MULTIPLY(self, inst):
        self.binary_op(operator.mul)

    def op_INPLACE_DIVIDE(self, inst):
        self.binary_op(operator.floordiv)

    def op_INPLACE_FLOOR_DIVIDE(self, inst):
        self.binary_op(operator.floordiv)

    def op_INPLACE_TRUE_DIVIDE(self, inst):
        self.binary_op(operator.truediv)

    def op_INPLACE_MODULO(self, inst):
        self.binary_op(operator.mod)

    def op_INPLACE_POWER(self, inst):
        self.binary_op(operator.pow)

    def op_INPLACE_RSHIFT(self, inst):
        self.binary_op(operator.rshift)

    def op_INPLACE_LSHIFT(self, inst):
        self.binary_op(operator.lshift)

    def op_INPLACE_AND(self, inst):
        self.binary_op(operator.and_)

    def op_INPLACE_OR(self, inst):
        self.binary_op(operator.or_)

    def op_INPLACE_XOR(self, inst):
        self.binary_op(operator.xor)

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
            if not ps:
                p = set()
            else:
                p = reduce(set.intersection, ps)
            new = set([blk]) | p
            if new != d:
                doms[blk] = new
                changed = True

    return doms

class RenameDefinition(object):
    '''Rename definition so that each assignment is unique during it is defined
    along the backbone --- the backbone is the controlflow path that must
    be visited.
    '''
    def __init__(self, blocks, doms, backbone):
        self.blocks = blocks
        self.doms = doms
        self.backbone = backbone

        self.defcount = defaultdict(int)
        self.defmap = {}
        for b in self.blocks:
            self.defmap[b] = {}

    def run(self):
        for b in self.blocks:
            for i in b.code:
                with error_context(lineno=i.lineno):
                    if i.opcode == 'load':
                        self.op_load(i)
                    elif i.opcode == 'store':
                        self.op_store(i)

    def op_load(self, load):
        name, _ = self.lookup(load.block, load.name)
        if not name:
            raise NameError('variable "%s" can be undefined' % load.name)
        load.update(name=name)

    def op_store(self, store):
        block = store.block
        name = store.name
        rename, defblock = self.lookup(store.block, name)
        if not rename or self.can_define(block) or defblock is block:
            rename = '%s.%d' % (name, self.defcount[name])
            self.defcount[name] += 1
            self.defmap[block][name] = rename
        store.update(name=rename)

    def can_define(self, block):
        return block in self.backbone

    def lookup(self, block, name):
        '''return (name, block) where block is the defined block
        '''
        doms = sorted((x.offset, x) for x in self.doms[block])
        for offset, block in reversed(doms):
            defmap = self.defmap[block]
            if name in defmap:
                return defmap[name], block
        return None, None

#---------------------------------------------------------------------------
# Internals

class BlockMap(object):
    def __init__(self):
        self._map = {}

    def __getitem__(self, offset):
        try:
            return self._map[offset]
        except KeyError:
            self._map[offset] = Block(offset)
            return self._map[offset]

    def __contains__(self, offset):
        return offset in self._map

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

    def last(self):
        return self.sorted()[-1][1]

class Incomings(object):
    def __init__(self):
        self.incomings = {}

    def add_incoming(self, block, value):
        assert block not in self.incomings, "duplicated incoming block for phi"
        self.incomings[block] = value

    def __repr__(self):
        ins = '; '.join('%r=%r' % it for it in self.incomings.iteritems())
        return '[%s]' % ins

    def __iter__(self):
        return self.incomings.iteritems()

    def values(self):
        return self.incomings.values()

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

    def is_terminated(self):
        return self.terminator is not None

    def is_dead(self):
        return not self.incoming_blocks and not self.outgoing_blocks

    def connect(self, nextblk):
        nextblk.incoming_blocks.add(self)
        self.outgoing_blocks.add(nextblk)

    def descr(self):
        def wrap(text, w=27):
            lines = text.splitlines()
            return '\n'.join([lines[0]] + [_indent(ln, w) for ln in lines[1:]])

        ins = ', '.join(str(b.offset) for b in self.incoming_blocks)
        head = ["block %4d        ; incoming %s" % (self.offset, ins)]
        body = ["    {!r:<20} = {!s}".format(c, wrap(str(c))) for c in self.code]
        tail = ["    %s\n" % wrap(str(self.terminator), w=4)]
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
        return [(k, getattr(self, k)) for k in self.attrs]

    def __str__(self):
        head = '%s ' % self.opcode
        w = len(head)
        if self.attrs:
            keys, vals = zip(*sorted(self.list_attrs()))
            max_key_len = max(len(k) for k in keys)
            keycolfmt = '{!s:<%d} = {!r}' % (max_key_len)
            rows = [keycolfmt.format(k, v) for k, v in zip(keys, vals)]
            out = '\n'.join([head + rows[0]] +
                            [_indent(r, w) for r in rows[1:]])
        else:
            out = head

        return out

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

def _indent(t, w, c=' '):
    ind = c * w
    return '%s%s' % (ind, t)
