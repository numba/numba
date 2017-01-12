from __future__ import print_function, division, absolute_import

import collections
from pprint import pprint
import sys
import warnings

from numba import utils


class DataFlowAnalysis(object):
    """
    Perform stack2reg

    This is necessary to resolve blocks that propagates stack value.
    This would allow the use of `and` and `or` and python2.6 jumps.
    """

    def __init__(self, cfa):
        self.cfa = cfa
        self.bytecode = cfa.bytecode
        # { block offset -> BlockInfo }
        self.infos = {}

    def run(self):
        for blk in self.cfa.iterliveblocks():
            self.infos[blk.offset] = self.run_on_block(blk)

    def run_on_block(self, blk):
        incoming_blocks = []
        info = BlockInfo(blk.offset, incoming_blocks)

        for ib, pops in self.cfa.incoming_blocks(blk):
            # By nature of Python bytecode, there will be no incoming
            # variables from subsequent blocks.  This is an easy way
            # of breaking the potential circularity of the problem.
            if ib.offset >= blk.offset:
                continue
            ib = self.infos[ib.offset]
            incoming_blocks.append(ib)

            # Compute stack offset at block entry
            # The stack effect of our predecessors should be known
            assert ib.stack_offset is not None, ib
            new_offset = ib.stack_offset + ib.stack_effect - pops
            if new_offset < 0:
                raise RuntimeError("computed negative stack offset for %s"
                                   % blk)
            if info.stack_offset is None:
                info.stack_offset = new_offset
            elif info.stack_offset != new_offset:
                warnings.warn("inconsistent stack offset for %s" % blk,
                              RuntimeWarning)

            # Compute syntax blocks at block entry
            assert ib.syntax_blocks is not None, ib
            if info.syntax_blocks is None:
                info.syntax_blocks = ib.syntax_blocks[:]
            elif info.syntax_blocks != ib.syntax_blocks:
                warnings.warn("inconsistent entry syntax blocks for %s" % blk,
                              RuntimeWarning)

        if info.stack_offset is None:
            # No incoming blocks => assume it's the entry block
            info.stack_offset = 0
            info.syntax_blocks = []
        info.stack_effect = 0

        for offset in blk:
            inst = self.bytecode[offset]
            self.dispatch(info, inst)
        return info

    def dump(self):
        for blk in utils.itervalues(self.infos):
            blk.dump()

    def dispatch(self, info, inst):
        fname = "op_%s" % inst.opname.replace('+', '_')
        fn = getattr(self, fname)
        fn(info, inst)

    def dup_topx(self, info, inst, count):
        orig = [info.pop() for _ in range(count)]
        orig.reverse()
        # We need to actually create new temporaries if we want the
        # IR optimization pass to work correctly (see issue #580)
        duped = [info.make_temp() for _ in range(count)]
        info.append(inst, orig=orig, duped=duped)
        for val in orig:
            info.push(val)
        for val in duped:
            info.push(val)

    def add_syntax_block(self, info, block):
        """
        Add an inner syntax block.
        """
        block.stack_offset = info.stack_offset
        info.syntax_blocks.append(block)

    def pop_syntax_block(self, info):
        """
        Pop the innermost syntax block and revert its stack effect.
        """
        block = info.syntax_blocks.pop()
        assert info.stack_offset >= block.stack_offset
        while info.stack_offset + info.stack_effect > block.stack_offset:
            info.pop(discard=True)
        return block

    def op_DUP_TOPX(self, info, inst):
        count = inst.arg
        assert 1 <= count <= 5, "Invalid DUP_TOPX count"
        self.dup_topx(info, inst, count)

    def op_DUP_TOP(self, info, inst):
        self.dup_topx(info, inst, count=1)

    def op_DUP_TOP_TWO(self, info, inst):
        self.dup_topx(info, inst, count=2)

    def op_ROT_TWO(self, info, inst):
        first = info.pop()
        second = info.pop()
        info.push(first)
        info.push(second)

    def op_ROT_THREE(self, info, inst):
        first = info.pop()
        second = info.pop()
        third = info.pop()
        info.push(first)
        info.push(third)
        info.push(second)

    def op_ROT_FOUR(self, info, inst):
        first = info.pop()
        second = info.pop()
        third = info.pop()
        forth = info.pop()
        info.push(first)
        info.push(forth)
        info.push(third)
        info.push(second)

    def op_UNPACK_SEQUENCE(self, info, inst):
        count = inst.arg
        iterable = info.pop()
        stores = [info.make_temp() for _ in range(count)]
        tupleobj = info.make_temp()
        info.append(inst, iterable=iterable, stores=stores, tupleobj=tupleobj)
        for st in reversed(stores):
            info.push(st)

    def op_BUILD_TUPLE(self, info, inst):
        count = inst.arg
        items = list(reversed([info.pop() for _ in range(count)]))
        tup = info.make_temp()
        info.append(inst, items=items, res=tup)
        info.push(tup)

    def op_BUILD_LIST(self, info, inst):
        count = inst.arg
        items = list(reversed([info.pop() for _ in range(count)]))
        lst = info.make_temp()
        info.append(inst, items=items, res=lst)
        info.push(lst)

    def op_BUILD_MAP(self, info, inst):
        dct = info.make_temp()
        count = inst.arg
        items = []
        if sys.version_info >= (3, 5):
            # In 3.5+, BUILD_MAP takes <count> pairs from the stack
            for i in range(count):
                v, k = info.pop(), info.pop()
                items.append((k, v))
        info.append(inst, items=items[::-1], size=count, res=dct)
        info.push(dct)

    def op_BUILD_SET(self, info, inst):
        count = inst.arg
        items = [info.pop() for _ in range(count)]
        res = info.make_temp()
        info.append(inst, items=items, res=res)
        info.push(res)

    def op_POP_TOP(self, info, inst):
        info.pop(discard=True)

    def op_STORE_ATTR(self, info, inst):
        target = info.pop()
        value = info.pop()
        info.append(inst, target=target, value=value)

    def op_DELETE_ATTR(self, info, inst):
        target = info.pop()
        info.append(inst, target=target)

    def op_STORE_FAST(self, info, inst):
        value = info.pop()
        info.append(inst, value=value)

    def op_STORE_MAP(self, info, inst):
        key = info.pop()
        value = info.pop()
        dct = info.tos
        info.append(inst, dct=dct, key=key, value=value)

    def op_LOAD_FAST(self, info, inst):
        name = self.bytecode.co_varnames[inst.arg]
        res = info.make_temp(name)
        info.append(inst, res=res)
        info.push(res)

    def op_LOAD_CONST(self, info, inst):
        res = info.make_temp('const')
        info.append(inst, res=res)
        info.push(res)

    def op_LOAD_GLOBAL(self, info, inst):
        res = info.make_temp()
        info.append(inst, res=res)
        info.push(res)

    def op_LOAD_DEREF(self, info, inst):
        res = info.make_temp()
        info.append(inst, res=res)
        info.push(res)

    def op_LOAD_ATTR(self, info, inst):
        item = info.pop()
        res = info.make_temp()
        info.append(inst, item=item, res=res)
        info.push(res)

    def op_BINARY_SUBSCR(self, info, inst):
        index = info.pop()
        target = info.pop()
        res = info.make_temp()
        info.append(inst, index=index, target=target, res=res)
        info.push(res)

    def op_STORE_SUBSCR(self, info, inst):
        index = info.pop()
        target = info.pop()
        value = info.pop()
        info.append(inst, target=target, index=index, value=value)

    def op_DELETE_SUBSCR(self, info, inst):
        index = info.pop()
        target = info.pop()
        info.append(inst, target=target, index=index)

    def op_GET_ITER(self, info, inst):
        value = info.pop()
        res = info.make_temp()
        info.append(inst, value=value, res=res)
        info.push(res)

    def op_FOR_ITER(self, info, inst):
        iterator = info.tos
        pair = info.make_temp()
        indval = info.make_temp()
        pred = info.make_temp()
        info.append(inst, iterator=iterator, pair=pair, indval=indval, pred=pred)
        info.push(indval)

    def _op_call_function(self, info, inst, has_vararg):
        narg = inst.arg & 0xff
        nkws = (inst.arg >> 8) & 0xff

        def pop_kws():
            val = info.pop()
            key = info.pop()
            return key, val

        vararg = info.pop() if has_vararg else None
        kws = list(reversed([pop_kws() for _ in range(nkws)]))
        args = list(reversed([info.pop() for _ in range(narg)]))
        func = info.pop()

        res = info.make_temp()
        info.append(inst, func=func, args=args, kws=kws, res=res,
                    vararg=vararg)
        info.push(res)

    def op_CALL_FUNCTION(self, info, inst):
        self._op_call_function(info, inst, has_vararg=False)

    def op_CALL_FUNCTION_VAR(self, info, inst):
        self._op_call_function(info, inst, has_vararg=True)

    def op_PRINT_ITEM(self, info, inst):
        warnings.warn("Python2 style print partially supported.  Please use "
                      "Python3 style print.", RuntimeWarning)
        item = info.pop()
        printvar = info.make_temp()
        res = info.make_temp()
        info.append(inst, item=item, printvar=printvar, res=res)

    def op_PRINT_NEWLINE(self, info, inst):
        printvar = info.make_temp()
        res = info.make_temp()
        info.append(inst, printvar=printvar, res=res)

    def _unaryop(self, info, inst):
        val = info.pop()
        res = info.make_temp()
        info.append(inst, value=val, res=res)
        info.push(res)

    op_UNARY_NEGATIVE = _unaryop
    op_UNARY_POSITIVE = _unaryop
    op_UNARY_NOT = _unaryop
    op_UNARY_INVERT = _unaryop

    def _binaryop(self, info, inst):
        rhs = info.pop()
        lhs = info.pop()
        res = info.make_temp()
        info.append(inst, lhs=lhs, rhs=rhs, res=res)
        info.push(res)

    op_COMPARE_OP = _binaryop

    op_INPLACE_ADD = _binaryop
    op_INPLACE_SUBTRACT = _binaryop
    op_INPLACE_MULTIPLY = _binaryop
    op_INPLACE_DIVIDE = _binaryop
    op_INPLACE_TRUE_DIVIDE = _binaryop
    op_INPLACE_FLOOR_DIVIDE = _binaryop
    op_INPLACE_MODULO = _binaryop
    op_INPLACE_POWER = _binaryop
    op_INPLACE_MATRIX_MULTIPLY = _binaryop

    op_INPLACE_LSHIFT = _binaryop
    op_INPLACE_RSHIFT = _binaryop
    op_INPLACE_AND = _binaryop
    op_INPLACE_OR = _binaryop
    op_INPLACE_XOR = _binaryop

    op_BINARY_ADD = _binaryop
    op_BINARY_SUBTRACT = _binaryop
    op_BINARY_MULTIPLY = _binaryop
    op_BINARY_DIVIDE = _binaryop
    op_BINARY_TRUE_DIVIDE = _binaryop
    op_BINARY_FLOOR_DIVIDE = _binaryop
    op_BINARY_MODULO = _binaryop
    op_BINARY_POWER = _binaryop
    op_BINARY_MATRIX_MULTIPLY = _binaryop

    op_BINARY_LSHIFT = _binaryop
    op_BINARY_RSHIFT = _binaryop
    op_BINARY_AND = _binaryop
    op_BINARY_OR = _binaryop
    op_BINARY_XOR = _binaryop

    def op_SLICE_0(self, info, inst):
        """
        TOS = TOS[:]
        """
        tos = info.pop()
        res = info.make_temp()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos, res=res, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)
        info.push(res)

    def op_SLICE_1(self, info, inst):
        """
        TOS = TOS1[TOS:]
        """
        tos = info.pop()
        tos1 = info.pop()
        res = info.make_temp()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, start=tos, res=res, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)
        info.push(res)

    def op_SLICE_2(self, info, inst):
        """
        TOS = TOS1[:TOS]
        """
        tos = info.pop()
        tos1 = info.pop()
        res = info.make_temp()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, stop=tos, res=res, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)
        info.push(res)

    def op_SLICE_3(self, info, inst):
        """
        TOS = TOS2[TOS1:TOS]
        """
        tos = info.pop()
        tos1 = info.pop()
        tos2 = info.pop()
        res = info.make_temp()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        info.append(inst, base=tos2, start=tos1, stop=tos, res=res,
                    slicevar=slicevar, indexvar=indexvar)
        info.push(res)

    def op_STORE_SLICE_0(self, info, inst):
        """
        TOS[:] = TOS1
        """
        tos = info.pop()
        value = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos, value=value, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_1(self, info, inst):
        """
        TOS1[TOS:] = TOS2
        """
        tos = info.pop()
        tos1 = info.pop()
        value = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, start=tos, slicevar=slicevar,
                    value=value, indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_2(self, info, inst):
        """
        TOS1[:TOS] = TOS2
        """
        tos = info.pop()
        tos1 = info.pop()
        value = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, stop=tos, value=value, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_3(self, info, inst):
        """
        TOS2[TOS1:TOS] = TOS3
        """
        tos = info.pop()
        tos1 = info.pop()
        tos2 = info.pop()
        value = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        info.append(inst, base=tos2, start=tos1, stop=tos, value=value,
                    slicevar=slicevar, indexvar=indexvar)

    def op_DELETE_SLICE_0(self, info, inst):
        """
        del TOS[:]
        """
        tos = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_1(self, info, inst):
        """
        del TOS1[TOS:]
        """
        tos = info.pop()
        tos1 = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, start=tos, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_2(self, info, inst):
        """
        del TOS1[:TOS]
        """
        tos = info.pop()
        tos1 = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        nonevar = info.make_temp()
        info.append(inst, base=tos1, stop=tos, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_3(self, info, inst):
        """
        del TOS2[TOS1:TOS]
        """
        tos = info.pop()
        tos1 = info.pop()
        tos2 = info.pop()
        slicevar = info.make_temp()
        indexvar = info.make_temp()
        info.append(inst, base=tos2, start=tos1, stop=tos,
                    slicevar=slicevar, indexvar=indexvar)

    def op_BUILD_SLICE(self, info, inst):
        """
        slice(TOS1, TOS) or slice(TOS2, TOS1, TOS)
        """
        argc = inst.arg
        if argc == 2:
            tos = info.pop()
            tos1 = info.pop()
            start = tos1
            stop = tos
            step = None
        elif argc == 3:
            tos = info.pop()
            tos1 = info.pop()
            tos2 = info.pop()
            start = tos2
            stop = tos1
            step = tos
        else:
            raise Exception("unreachable")
        slicevar = info.make_temp()
        res = info.make_temp()
        info.append(inst, start=start, stop=stop, step=step, res=res,
                    slicevar=slicevar)
        info.push(res)

    def op_POP_JUMP_IF_TRUE(self, info, inst):
        pred = info.pop()
        info.append(inst, pred=pred)
        info.terminator = inst

    def op_POP_JUMP_IF_FALSE(self, info, inst):
        pred = info.pop()
        info.append(inst, pred=pred)
        info.terminator = inst

    def op_JUMP_IF_TRUE(self, info, inst):
        pred = info.tos
        info.append(inst, pred=pred)
        info.terminator = inst

    def op_JUMP_IF_FALSE(self, info, inst):
        pred = info.tos
        info.append(inst, pred=pred)
        info.terminator = inst

    op_JUMP_IF_FALSE_OR_POP = op_JUMP_IF_FALSE
    op_JUMP_IF_TRUE_OR_POP = op_JUMP_IF_TRUE

    def op_JUMP_ABSOLUTE(self, info, inst):
        info.append(inst)
        info.terminator = inst

    def op_JUMP_FORWARD(self, info, inst):
        info.append(inst)
        info.terminator = inst

    def op_BREAK_LOOP(self, info, inst):
        self.pop_syntax_block(info)
        info.append(inst)
        info.terminator = inst

    def op_RETURN_VALUE(self, info, inst):
        info.append(inst, retval=info.pop(), castval=info.make_temp())
        info.terminator = inst

    def op_YIELD_VALUE(self, info, inst):
        val = info.pop()
        res = info.make_temp()
        info.append(inst, value=val, res=res)
        info.push(res)

    def op_SETUP_LOOP(self, info, inst):
        self.add_syntax_block(info, LoopBlock())
        info.append(inst)

    def op_POP_BLOCK(self, info, inst):
        block = self.pop_syntax_block(info)
        info.append(inst)

    def op_RAISE_VARARGS(self, info, inst):
        if inst.arg == 0:
            exc = None
        elif inst.arg == 1:
            exc = info.pop()
        else:
            raise ValueError("Multiple argument raise is not supported.")
        info.append(inst, exc=exc)

    def _ignored(self, info, inst):
        pass


class LoopBlock(object):
    __slots__ = ('stack_offset',)

    def __init__(self):
        self.stack_offset = None


class BlockInfo(object):
    def __init__(self, offset, incoming_blocks):
        self.offset = offset
        # The list of incoming BlockInfo objects (obtained by control
        # flow analysis).
        self.incoming_blocks = incoming_blocks
        self.stack = []
        # Outgoing variables from this block:
        #   { outgoing phi name -> var name }
        self.outgoing_phis = {}
        self.insts = []
        self.tempct = 0
        self._term = None
        self.stack_offset = None
        self.stack_effect = 0
        self.syntax_blocks = None

    def __repr__(self):
        return "<%s at offset %d>" % (self.__class__.__name__, self.offset)

    def dump(self):
        print("offset", self.offset, "{")
        print("  stack: ", end='')
        pprint(self.stack)
        pprint(self.insts)
        print("}")

    def make_temp(self, prefix=''):
        self.tempct += 1
        name = '$%s%s.%s' % (prefix, self.offset, self.tempct)
        return name

    def push(self, val):
        self.stack_effect += 1
        self.stack.append(val)

    def pop(self, discard=False):
        """
        Pop a variable from the stack, or request it from incoming blocks if
        the stack is empty.
        If *discard* is true, the variable isn't meant to be used anymore,
        which allows reducing the number of temporaries created.
        """
        if not self.stack:
            self.stack_offset -= 1
            if not discard:
                return self.make_incoming()
        else:
            self.stack_effect -= 1
            return self.stack.pop()

    def make_incoming(self):
        """
        Create an incoming variable (due to not enough values being
        available on our stack) and request its assignment from our
        incoming blocks' own stacks.
        """
        assert self.incoming_blocks
        ret = self.make_temp('phi')
        for ib in self.incoming_blocks:
            stack_index = self.stack_offset + self.stack_effect
            ib.request_outgoing(self, ret, stack_index)
        return ret

    def request_outgoing(self, outgoing_block, phiname, stack_index):
        """
        Request the assignment of the next available stack variable
        for block *outgoing_block* with target name *phiname*.
        """
        if phiname in self.outgoing_phis:
            # If phiname was already requested, ignore this new request
            # (can happen with a diamond-shaped block flow structure).
            return
        if stack_index < self.stack_offset:
            assert self.incoming_blocks
            for ib in self.incoming_blocks:
                ib.request_outgoing(self, phiname, stack_index)
        else:
            varname = self.stack[stack_index - self.stack_offset]
            self.outgoing_phis[phiname] = varname

    @property
    def tos(self):
        r = self.pop()
        self.push(r)
        return r

    def append(self, inst, **kws):
        self.insts.append((inst.offset, kws))

    @property
    def terminator(self):
        assert self._term is None
        return self._term

    @terminator.setter
    def terminator(self, inst):
        self._term = inst
