from __future__ import print_function
from pprint import pprint


class DataFlowAnalysis(object):
    """
    Perform stack2reg

    This is necessary to resolve blocks that propagates stack value.
    This would allow the use of `and` and `or` and python2.6 jumps.
    """
    def __init__(self, cfa):
        self.cfa = cfa
        self.bytecode = cfa.bytecode
        self.infos = {}
        self.syntax_blocks = []

    def run(self):
        for blk in self.cfa.iterblocks():
            self.infos[blk.offset] = self.run_on_block(blk)

    def run_on_block(self, blk):
        info = BlockInfo(blk.offset)
        for offset in blk:
            inst = self.bytecode[offset]
            self.dispatch(info, inst)
        return info

    def dump(self):
        for blk in self.infos.itervalues():
            blk.dump()

    def dispatch(self, info, inst):
        fname = "op_%s" % inst.opname.replace('+', '_')
        fn = getattr(self, fname)
        fn(info, inst)

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

    def op_POP_TOP(self, info, inst):
        info.pop()

    def op_STORE_FAST(self, info, inst):
        value = info.pop()
        info.append(inst, value=value)

    def op_LOAD_FAST(self, info, inst):
        name = self.bytecode.code.co_varnames[inst.arg]
        info.push(name)

    def op_LOAD_CONST(self, info, inst):
        res = info.make_temp()
        info.append(inst, res=res)
        info.push(res)

    def op_LOAD_GLOBAL(self, info, inst):
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

    def op_GET_ITER(self, info, inst):
        value = info.pop()
        res = info.make_temp()
        info.append(inst, value=value, res=res)
        info.push(res)
        if self.syntax_blocks:
            loop = self.syntax_blocks[-1]
            if isinstance(loop, LoopBlock) and loop.iterator is None:
                loop.iterator = res

    def op_FOR_ITER(self, info, inst):
        loop = self.syntax_blocks[-1]
        iterator = loop.iterator
        indval = info.make_temp()
        pred = info.make_temp()
        info.append(inst, iterator=iterator, indval=indval, pred=pred)
        info.push(indval)

    def op_CALL_FUNCTION(self, info, inst):
        narg = inst.arg & 0xff
        nkws = (inst.arg >> 8) & 0xff

        def pop_kws():
            val = info.pop()
            key = info.pop()
            return key, val

        kws = list(reversed([pop_kws() for _ in range(nkws)]))
        args = list(reversed([info.pop() for _ in range(narg)]))
        func = info.pop()

        res = info.make_temp()
        info.append(inst, func=func, args=args, kws=kws, res=res)
        info.push(res)

    def op_UNARY_NEGATIVE(self, info, inst):
        val = info.pop()
        res = info.make_temp()
        info.append(inst, value=val, res=res)
        info.push(res)

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

    op_BINARY_ADD = _binaryop
    op_BINARY_SUBTRACT = _binaryop
    op_BINARY_MULTIPLY = _binaryop
    op_BINARY_DIVIDE = _binaryop
    op_BINARY_TRUE_DIVIDE = _binaryop
    op_BINARY_FLOOR_DIVIDE = _binaryop
    op_BINARY_MODULO = _binaryop
    op_BINARY_POWER = _binaryop

    def op_SLICE_3(self, info, inst):
        """
        TOS = TOS2[TOS1:TOS]
        """
        tos = info.pop()
        tos1 = info.pop()
        tos2 = info.pop()
        res = info.make_temp()
        info.append(inst, base=tos2, start=tos1, stop=tos, res=res)
        info.push(res)

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

    def op_RETURN_VALUE(self, info, inst):
        info.append(inst, retval=info.pop())
        info.terminator = inst

    def op_SETUP_LOOP(self, info, inst):
        self.syntax_blocks.append(LoopBlock())
        info.append(inst)

    def op_POP_BLOCK(self, info, inst):
        block = self.syntax_blocks.pop()
        if isinstance(block, LoopBlock):
            info.append(inst, delitem=block.iterator)
        else:
            info.append(inst)

    def _ignored(self, info, inst):
        pass


class LoopBlock(object):
    __slots__ = 'iterator'

    def __init__(self):
        self.iterator = None


class BlockInfo(object):
    def __init__(self, offset):
        self.offset = offset
        self.stack = []
        self.incomings = []
        self.insts = []
        self.tempct = 0
        self._term = None

    def dump(self):
        print("offset", self.offset, "{")
        print("  stack: ", end='')
        pprint(self.stack)
        print("  incomings: ", end='')
        pprint(self.incomings)
        pprint(self.insts)
        print("}")

    def make_temp(self):
        self.tempct += 1
        name = '$%d.%d' % (self.offset, self.tempct)
        return name

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        # TODO: lingering incoming values
        if not self.stack:
            assert not self.insts
            ret = self.make_temp()
            self.incomings.append(ret)
        else:
            ret = self.stack.pop()
        return ret

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
