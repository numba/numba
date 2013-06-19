import __builtin__
import dis
import operator
import inspect
from collections import defaultdict, namedtuple

from .bytecode import ByteCode, BYTECODE_TABLE
from .utils import SortedMap
from .errors import CompileError

Arg = namedtuple('Arg', ['num', 'name'])
Call = namedtuple('Call', ['func', 'args', 'kws'])
Global = namedtuple('Global', ['name', 'value'])
Phi = namedtuple('Phi', ['name', 'incomings'])
Const = namedtuple('Const', ['value'])
For = namedtuple('For', ['index', 'stop', 'step'])
BinOp = namedtuple('BinOp', ['lhs', 'rhs'])
BoolOp = namedtuple('BoolOp', ['lhs', 'rhs'])
UnaryOp = namedtuple('UnaryOp', ['value'])
Unpack = namedtuple('Unpack', ['obj', 'offset'])
GetItem = namedtuple('GetItem', ['obj', 'idx'])
SetItem = namedtuple('SetItem', ['obj', 'idx', 'value'])

ComplexGetAttr = namedtuple('ComplexGetAttr', ['obj', 'attr'])

ArrayAttr = namedtuple('ArrayAttr', ['obj', 'attr', 'idx'])

Jump = namedtuple('Jump', ['target'])
Branch = namedtuple('Branch', ['cmp', 'false', 'true'])
Ret = namedtuple('Ret', ['value'])


'''MEMORY_OP

A set of expression kind that is memory store expression.
These expressions is pardoned by the strip used.
'''
MEMORY_OP = frozenset(['SetItem'])
SIDE_EFFECT = MEMORY_OP | set(['Call'])

BINARYOP_MAP = {
     '+': 'BinOp',
     '-': 'BinOp',
     '*': 'BinOp',
     '/': 'BinOp',
    '//': 'BinOp',
     '%': 'BinOp',
     '&': 'BinOp',
     '|': 'BinOp',
     '^': 'BinOp',
'ForInit': 'BinOp',
}

BOOLOP_MAP = {
      '>': 'BoolOp',
     '>=': 'BoolOp',
      '<': 'BoolOp',
     '<=': 'BoolOp',
     '==': 'BoolOp',
     '!=': 'BoolOp',
}

UNARYOP_MAP = {
    '~': 'UnaryOp',
}

OP_MAP = dict()
OP_MAP.update(BINARYOP_MAP)
OP_MAP.update(UNARYOP_MAP)
OP_MAP.update(BOOLOP_MAP)

class TranslateError(CompileError):
    def __init__(self, inst, msg):
        super(TranslateError, self).__init__(inst.lineno, msg)

PendingFor = namedtuple('PendingFor', ['prev', 'block', 'itername', 'iter'])

def get_global_value(func, name):
    sep = name.split('.')
    prefix = sep[0]
    try:
        val = func.func_globals[prefix]
    except KeyError:
        val = getattr(__builtin__, prefix)

    for attr in sep[1:]:
        val = getattr(val, attr)
    return val

class SymbolicExecution(object):
    def __init__(self, func):
        self.func = func

        # bytecode info
        self.bytecode = ByteCode(func)
        self.varnames = self.bytecode.code.co_varnames
        self.consts = self.bytecode.code.co_consts
        self.names = self.bytecode.code.co_names

        # symoblic execution info
        self.valuemap = {}
        self.stack = []
        self.scopes = []
        self.blocks = dict((offset, Block(offset, self.valuemap))
                           for offset in self.bytecode.labels)
        self.ignore_first_pop_top = set()
        self.curblock = self.blocks[0]
        self.pending_for = []
        self.skip = 0

        # init
        self.prepare_arguements()

    def prepare_arguements(self):
        argspec = inspect.getargspec(self.func)
        assert not argspec.defaults
        assert not argspec.varargs
        assert not argspec.keywords
        for i, arg in enumerate(argspec.args):
            value = Expr('Arg', None, Arg(num=i, name=arg))
            self.curblock.varmap[arg].append(self.insert(value))
#        for var in self.varnames:
#            if var not in self.curblock.varmap:
#                undef = Expr('Undef', None)
#                self.curblock.varmap[var].append(self.insert(undef))

    def visit(self):
        for inst in self.bytecode:
            attr = 'visit_' + inst.opname
            func = getattr(self, attr, self.generic_visit)
            # detect current block
            oldblock = self.curblock
            self.curblock = self.blocks.get(inst.offset, self.curblock)

            if self.curblock is not oldblock:
                self.enter_new_block(oldblock)

            if self.curblock.terminator:
                continue
        
            if self.skip:
                self.skip -= 1
                continue

            # logic to skip the first POP_TOP for python2.6
            if self.curblock is not oldblock:
                if self.curblock.offset in self.ignore_first_pop_top:
                    if inst.opname == 'POP_TOP':
                        continue # skipped

            func(inst)

        assert not self.stack, self.stack
        self.complete_for()
        self.strip_dead_block()
        self.doms = find_dominators(self.blocks)
        self.idoms = find_immediate_dominator(self.blocks, self.doms)
        self.complete_phi_nodes()
        self.strip_unused()
    

    def strip_dead_block(self):
        removed = set()
        for blk in list(self.blocks.itervalues()):
            if blk.offset != 0 and not blk.incomings:
                removed.add(blk.offset)
                del self.blocks[blk.offset]

        for blk in self.blocks.itervalues():
            blk.incomings -= removed

    def complete_for(self):
        for pf in self.pending_for:
            prev = self.blocks[pf.prev]
            step = pf.iter.step or prev.insert(Expr('Const', None, Const(1)))
            start = pf.iter.start or prev.insert(Expr('Const', None, Const(0)))
            start = prev.insert(Expr('ForInit', None, BinOp(lhs=start, rhs=step)))
            prev.varmap[pf.itername].append(start)

    def complete_phi_nodes(self):
        changed = True
        while changed:
            changed = False
            for blk in self.blocks.itervalues():
                for val in blk.body:
                    if val.kind == 'Phi' and len(val.args.incomings) != len(blk.incomings):
                        phi = val
                        for inblkoff in blk.incomings:
                            inblk = self.blocks[inblkoff]
                            lastdefs = inblk.varmap[phi.args.name]
                            if lastdefs:
                                lastval = lastdefs[-1]
                            else:
                                # find the closest definition
                                lastval = self.find_def(inblkoff, phi.args.name)
                                if not lastval:
                                    raise TranslateError(val, "variable %s can be undef" % phi.args.name)
                                changed = True
                            phi.args.incomings.add((inblkoff, lastval))
                            if lastval:
                                update_uses(phi.ref, [lastval])
                    else:
                        break

    def find_def(self, blkoff, name):
        '''Find variable definition in the IDOM of a block
        '''
        if blkoff == 0: return
        idomoff = self.idoms[blkoff]
        idomblk = self.blocks[idomoff]
        if not idomblk.varmap[name]:
            res = self.find_def(idomoff, name)
        else:
            res = idomblk.varmap[name][-1]
        if res:
            blk = self.blocks[blkoff]
            phi = blk.get_last_def(name)
            return phi

    def strip_unused(self):
        for blk in self.blocks.itervalues():
            marked = set()
            for expr in blk.body:
                if not expr.ref.uses and expr.kind not in SIDE_EFFECT:
                    marked.add(expr)
            for expr in marked:
                blk.body.remove(expr)


    def generic_visit(self, inst):
        raise TranslateError(inst, "unsupported bytecode %s" % inst)

    def enter_new_block(self, oldblock):
        '''Stuff to do when entering a new block.

        NOTE: self.curblock is pointing to the new block.
        '''
        if oldblock.terminator is None:
            oldblock.terminator = Term('Jump', None, Jump(self.curblock.offset))
            oldblock.connect(self.curblock)

#        for name in self.varnames:
#            phi = self.insert(Expr('Phi', None,
#                              Phi(name=name, incomings=set())))
#            self.curblock.varmap[name].append(phi)

    def insert(self, expr):
        return self.curblock.insert(expr)

    def terminate(self, term):
        assert self.curblock.terminator is None
        self.curblock.terminator = term

    def make_subblock(self, offset):
        oldblock = self.curblock
        self.blocks[offset] = Block(offset, self.valuemap)

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        return self.stack.pop()

    def peek(self, offset):
        self.skip += 1
        return self.bytecode[offset]

    def dump(self):
        buf = []
        for blk in SortedMap(self.blocks.items()).itervalues():
            buf += [blk.dump()]
        return '\n'.join(buf)

    ### specialized visitors

    def visit_POP_TOP(self, inst):
        self.stack.pop()

    def visit_DUP_TOPX(self, inst):
        ct = inst.arg
        self.stack.extend(self.stack[-ct:])

    def visit_DUP_TOP(self, inst):
        self.stack.append(self.stack[-1])

    def visit_ROT_THREE(self, inst):
        tos = self.pop()
        second = self.pop()
        third = self.pop()
        self.push(tos)
        self.push(third)
        self.push(second)

    def visit_ROT_TWO(self, inst):
        one = self.pop()
        two = self.pop()
        self.push(two)
        self.push(one)

    def visit_LOAD_CONST(self, inst):
        val = self.consts[inst.arg]
        expr = Expr('Const', inst, Const(value=val))
        self.push(self.insert(expr))

    def visit_LOAD_GLOBAL(self, inst):
        name = self.names[inst.arg]
        expr = Expr('Global', inst, Global(name=name, value=None))
        self.push(self.insert(expr))

    def visit_STORE_FAST(self, inst):
        name = self.varnames[inst.arg]
        self.curblock.varmap[name].append(self.pop())

    def visit_LOAD_FAST(self, inst):
        name = self.varnames[inst.arg]
        defn = self.curblock.get_last_def(name)
#        if defs[-1].value.kind == 'Undef':
#            raise TranslateError(inst, 'variable %s is not defined' % name)
        self.push(defn)

    def visit_CALL_FUNCTION(self, inst):
        argc = inst.arg & 0xff
        kwsc = (inst.arg >> 8) & 0xff

        kws = list(reversed([self.pop() for i in range(kwsc)]))
        args = list(reversed([self.pop() for i in range(argc)]))

        func = self.pop()

        if func.value.kind != 'Global':
            msg = "can only call global functions"
            raise TranslateError(inst, msg)

        funcname = func.value.args[0]

        funcobj = None
        if (funcname in ['range', 'xrange']
                and funcname not in self.func.func_globals):
            funcobj = get_global_value(self.func, funcname)

        if funcobj in [range, xrange]:
            if kws:
                msg = "range/xrange do not accept keywords"
                raise TranslateError(inst, msg)
            if len(args) not in [1, 2, 3]:
                msg = "invalid # of args to range/xrange"
                raise TranslateError(inst, msg)
            self.push(Range(args))
        else:
            expr = Expr('Call', inst, Call(func=funcname, args=args, kws=kws))
            self.push(self.insert(expr))

    def visit_GET_ITER(self, inst):
        obj = self.pop()
        if not isinstance(obj, Range):
            msg = "can only get iterator to range/xrange"
            raise TranslateError(inst, msg)
        self.push(obj)

    def visit_FOR_ITER(self, inst):
        iter = self.pop()
        if not isinstance(iter, Range):
            msg = "for loop must loop over range/xrange"
            raise TranslateError(inst, msg)

        storefast = self.peek(inst.next)

        if storefast.opname != 'STORE_FAST':
            raise TranslateError(storefast, 'unexpected for loop pattern')
        itername = self.varnames[storefast.arg]

        step = iter.step or self.insert(Expr('Const', inst, Const(value=1)))
        index0 = self.curblock.get_last_def(itername) # varmap[itername][-1]
        index = self.insert(Expr('+', inst, BinOp(lhs=index0, rhs=step)))
        self.curblock.varmap[itername].append(index)

        cmp = self.insert(Expr('For', inst, For(index=index, stop=iter.stop,
                                                step=step)))
        endloop = inst.next + inst.arg

        self.terminate(Term('Branch', inst,
                            Branch(cmp=cmp, false=endloop, true=inst.next)))

        pf = PendingFor(prev=self.scopes[-1][0],
                        block=self.curblock,
                        itername=itername,
                        iter=iter)

        self.pending_for.append(pf)

        blk = self.curblock
        self.make_subblock(inst.next)
        blk.connect(self.blocks[inst.next])
        blk.connect(self.blocks[endloop])

    def visit_RETURN_VALUE(self, inst):
        ref = self.pop()
        value = ref.value
        if value.kind == 'Const' and value.args[0] is None:
            self.terminate(Term('RetVoid', inst))
        else:
            self.terminate(Term('Ret', inst, Ret(ref)))

    def visit_JUMP_IF_FALSE(self, inst):
        truebr = inst.next
        falsebr = inst.next + inst.arg
        self.ignore_first_pop_top.add(truebr)
        self.ignore_first_pop_top.add(falsebr)
        cmp = self.pop()
        self._pop_jump_if_false(inst, cmp, truebr, falsebr)

    def visit_POP_JUMP_IF_FALSE(self, inst):
        truebr = inst.next
        falsebr = inst.arg
        cmp = self.pop()
        self._pop_jump_if_false(inst, cmp, truebr, falsebr)

    def _pop_jump_if_false(self, inst, cmp, truebr, falsebr):
        term = Term('Branch', inst,
                    Branch(cmp=cmp, true=truebr, false=falsebr))
        self.terminate(term)
        blk = self.curblock
        self.make_subblock(inst.next)
        blk.connect(self.blocks[truebr])
        blk.connect(self.blocks[falsebr])

    def visit_JUMP_ABSOLUTE(self, inst):
        term = Term('Jump', inst, Jump(inst.arg))
        self.terminate(term)
        self.curblock.connect(self.blocks[inst.arg])

    def visit_JUMP_FORWARD(self, inst):
        target = inst.arg + inst.next
        term = Term('Jump', inst, Jump(target))
        self.terminate(term)
        self.curblock.connect(self.blocks[target])

    def visit_SETUP_LOOP(self, inst):
        self.scopes.append((self.curblock.offset, inst.arg + inst.next))

    def visit_POP_BLOCK(self, inst):
        self.scopes.pop()

    def visit_BREAK_LOOP(self, inst):
        target = self.scopes[-1][1]
        term = Term('Jump', inst, Jump(target))
        self.terminate(term)
        self.curblock.connect(self.blocks[target])

    def visit_LOAD_ATTR(self, inst):
        ref = self.pop()
        gv = ref.value
        attr = self.names[inst.arg]
        if gv.kind == 'Global':
            gv.replace(name=('%s.%s' % (gv.args[0], attr)))
            self.push(ref)
        elif attr in ('imag', 'real'):
            expr = self.insert(Expr('ComplexGetAttr', inst,
                                    ComplexGetAttr(ref, attr)))
            self.push(expr)
        elif attr in ('ndim', 'size', 'shape', 'strides'):
            if attr in ['ndim', 'size']:
                item = self.insert(Expr('ArrayAttr', inst,
                                        ArrayAttr(ref, attr, None)))
            else:
                item  = ArrayAttr(ref, attr, None)
            self.push(item)
        else:
            msg = ('can only get attribute from globals, '
                   'complex numbers or arrays')
            raise TranslateError(inst, msg)


    def visit_BINARY_SUBSCR(self, inst):
        idx = self.pop()
        obj = self.pop()
        if isinstance(obj, ArrayAttr):
            # get item on array attr
            obj = obj._replace(idx=idx)
            self.push(self.insert(Expr('ArrayAttr', inst, obj)))
        else:
            # get item on array
            if not isinstance(idx, tuple):
                idx = (idx,)
            expr = Expr('GetItem', inst, GetItem(obj, idx))
            self.push(self.insert(expr))

    def visit_STORE_SUBSCR(self, inst):
        idx = self.pop()
        obj = self.pop()
        val = self.pop()
        
        if not isinstance(idx, tuple):
            idx = (idx,)
        self.insert(Expr('SetItem', inst, SetItem(obj, idx, val)))

    def visit_BUILD_TUPLE(self, inst):
        ct = inst.arg
        self.push(tuple(reversed([self.pop() for i in range(ct)])))
    
    def visit_UNPACK_SEQUENCE(self, inst):
        tos = self.pop()
        if tos.value.kind != 'Call':
            raise TranslateError(inst, "can only unpack from return value")
        count = inst.arg
        for i in reversed(range(count)):
            ref = self.insert(Expr('Unpack', inst, Unpack(tos, i)))
            self.push(ref)

    def visit_generic_binary(self, op, inst):
        rhs = self.pop()
        lhs = self.pop()
        expr = Expr(op, inst, BinOp(lhs=lhs, rhs=rhs))
        self.push(self.insert(expr))

    def visit_BINARY_ADD(self, inst):
        self.visit_generic_binary('+', inst)

    def visit_BINARY_SUBTRACT(self, inst):
        self.visit_generic_binary('-', inst)

    def visit_BINARY_MULTIPLY(self, inst):
        self.visit_generic_binary('*', inst)

    def visit_BINARY_DIVIDE(self, inst):
        self.visit_generic_binary('/', inst)

    def visit_BINARY_FLOOR_DIVIDE(self, inst):
        self.visit_generic_binary('//', inst)

    def visit_BINARY_MODULO(self, inst):
        self.visit_generic_binary('%', inst)

    def visit_BINARY_AND(self, inst):
        self.visit_generic_binary('&', inst)

    def visit_BINARY_OR(self, inst):
        self.visit_generic_binary('|', inst)

    def visit_BINARY_XOR(self, inst):
        self.visit_generic_binary('^', inst)

    def visit_COMPARE_OP(self, inst):
        rhs = self.pop()
        lhs = self.pop()
        op = dis.cmp_op[inst.arg]
        expr = Expr(op, inst, BoolOp(lhs=lhs, rhs=rhs))
        self.push(self.insert(expr))

    visit_generic_inplace = visit_generic_binary

    def visit_INPLACE_ADD(self, inst):
        self.visit_generic_inplace('+', inst)

    def visit_INPLACE_SUBTRACT(self, inst):
        self.visit_generic_inplace('-', inst)

    def visit_INPLACE_MULTIPLY(self, inst):
        self.visit_generic_inplace('*', inst)

    def visit_INPLACE_DIVIDE(self, inst):
        self.visit_generic_inplace('/', inst)

    def visit_INPLACE_FLOOR_DIVIDE(self, inst):
        self.visit_generic_inplace('//', inst)

    def visit_generic_unary(self, op, inst):
        tos = self.pop()
        expr = Expr(op, inst, UnaryOp(tos))
        self.push(self.insert(expr))

    def visit_UNARY_INVERT(self, inst):
        self.visit_generic_unary('~', inst)


#############################################################################
# Expressions

class Ref(object):
    def __init__(self, valuemap, value):
        self.valuemap = valuemap
        self.id = len(self.valuemap)
        self.valuemap[self] = value
        value.ref = self
        self.uses = set()

    @property
    def value(self):
        return self.valuemap[self]

    def __repr__(self):
        return '%%%s' % self.id

class Range(object):
    def __init__(self, args):
        self.start = None
        self.step = None
        if len(args) == 1:
            self.stop = args[0]
        elif len(args) == 2:
            self.start, self.stop = args
        elif len(args) == 3:
            self.start, self.stop, self.step = args

class Block(object):
    def __init__(self, offset, valuemap):
        self.offset = offset
        self.valuemap = valuemap
        self.body = []
        self.terminator = None
        self.edges = set()
        self.incomings = set()
        self.varmap = defaultdict(list)

    def connect(self, target):
        self.edges.add(target.offset)
        target.incomings.add(self.offset)

    def insert(self, value):
        self.body.append(value)
        return Ref(self.valuemap, value)

    def get_last_def(self, name):
        if not self.varmap[name]:
            phi = Expr('Phi', None, Phi(name, set()))
            self.body.insert(0, phi)
            self.varmap[name].append(Ref(self.valuemap, phi))
        return self.varmap[name][-1]

    def dump(self):
        indent = ' ' * 4
        buf = []
        edges = ','.join(sorted(map(str, self.edges)))
        incomings = ','.join(sorted(map(str, self.incomings)))
        buf += ['%s;     edges %s' % (indent, edges)]
        buf += ['%s; incomings %s' % (indent, incomings)]
        for expr in self.body:
            buf += ['%s%s = %s' % (indent, expr.ref, expr)]
        else:
            buf += ['%s%s' % (indent, self.terminator)]
        head = '%d:' % self.offset
        body = '\n'.join(buf)
        return '\n'.join([head, body])

    def __repr__(self):
        return 'Block(%d)' % self.offset

class Expr(object):
    def __init__(self, kind, inst, args=None):
        assert args is None or isinstance(args, tuple)
        self.kind = kind
        self.lineno = inst.lineno if inst is not None else -1
        self.args = args
        self.ref = None
        if args is not None:
            update_uses(self, args)

    def replace(self, **kws):
        self.args = self.args._replace(**kws)

    def __repr__(self):
        return '%s %s' % (self.kind, self.args)


class Term(object):
    def __init__(self, kind, inst, args=None):
        assert args is None or isinstance(args, tuple)
        self.kind = kind
        self.lineno = inst.lineno if inst is not None else -1
        self.args = args
        if args is not None:
            update_uses(self, args)

    def __repr__(self):
        return '%s %s' % (self.kind, self.args)

def update_uses(referee, vals):
    referee = referee.value if isinstance(referee, Ref) else referee
    for v in vals:
        if isinstance(v, Ref):
            v.uses.add(referee)
        elif isinstance(v, (tuple, list)):
            update_uses(referee, v)

##############################################################################
# Pass

def find_dominators(blocks):
    doms = defaultdict(set)
    doms[0].add(0)
    allblks = set(blocks.keys())

    remainblks = frozenset(blk for blk in blocks.itervalues() if blk.offset != 0)
    for blk in remainblks:
        doms[blk.offset] |= allblks

    changed = True
    while changed:
        changed = False
        for blk in remainblks:
            d = doms[blk.offset]
            ps = [doms[p] for p in blk.incomings]
            p = reduce(set.intersection, ps)
            new = set([blk.offset]) | p
            if new != d:
                doms[blk.offset] = new
                changed = True
    return doms

def block_hop(blocks, base, other):
    space = set(map(blocks.get, base.incomings))
    dist = 1
    # breath-first search
    while space and other not in space:
        new = set()
        for i in space:
            new |= set(map(blocks.get, i.incomings))
        space = new
        dist += 1
    return dist

def find_immediate_dominator(blocks, doms):
    idoms = {}
    for blk in blocks.itervalues():
        if blk.offset != 0: # ignore entry node
            dist = [(block_hop(blocks, blk, blocks[dom]), dom)
                    for dom in doms[blk.offset]
                    if dom is not blk.offset]
            if dist:
                idoms[blk.offset] = min(dist)[1]
    return idoms

def find_dominator_frontiers(blocks, doms, idoms):
    frontiers = defaultdict(set)
    for blk in blocks.itervalues():
        if len(blk.incomings) >= 2:
            fr = frontiers[blk.offset]
            for inc in map(blocks.get, blk.incomings):
                runner = inc
                while runner.offset != idoms[blk.offset]:
                    frontiers[runner.offset].add(blk.offset)
                    runner = blocks.get(idoms[runner.offset])
    return frontiers

def load_store_elmination(varmap, valuemap, blocks):
    doms = find_dominators(blocks)
    idoms = find_immediate_dominator(blocks, doms)
    assert False
    frontiers = find_dominator_frontiers(blocks, doms, idoms)
    philoc = reduce(set.union, frontiers.itervalues())
    incvals = defaultdict(lambda:defaultdict(set))
    incvals_entry = incvals[blocks[0]]
#    for

#    for blk in blocks.itervalues():
#        for var in varmap:
#            for next in blk.edges:
#                incvals[next][var].add(


#    # name -> block -> list of defs
#    defmap = defaultdict(lambda:defaultdict(list))
#    # name -> block -> list of uses
#    usemap = defaultdict(lambda:defaultdict(list))
#    for blk in blocks.itervalues():
#        for i, value in enumerate(blk.body):
#            if value.kind == 'Load':
#                usemap[value.args[0]][blk].append((i, value))
#            elif value.kind == 'Store':
#                defmap[value.args[0]][blk].append((i, value))

