"""
Implement python 3.8+ bytecode analysis
"""
import logging
import dis
from collections import namedtuple, defaultdict
from pprint import pformat

from numba.utils import UniqueDict
from numba.controlflow import NEW_BLOCKERS


_logger = logging.getLogger(__name__)



class Flow(object):
    def __init__(self, bytecode):
        self._bytecode = bytecode
        self.block_infos = UniqueDict()

    def run(self):
        firststate = State(bytecode=self._bytecode, pc=0, nstack=0)
        runner = Runner()
        runner.pending.append(firststate)

        first_encounter = UniqueDict()
        while runner.pending:
            _logger.debug('pending: %s', runner.pending)
            state = runner.pending.pop()
            if state not in runner.finished:
                first_encounter[state.pc_initial] = state
                while True:
                    runner.dispatch(state)
                    if state.has_terminated():
                        break
                    else:
                        state.advance_pc()
                        if self._is_implicit_new_block(state):
                            state.split_new_block()
                            break
                _logger.debug("end state. edges=%s", state.outgoing_edges)
                runner.finished.add(state)
                out_states = state.get_outgoing_states()
                runner.pending.extend(out_states)
        # Post process
        for state in sorted(runner.finished, key=lambda x: x.pc_initial):
            self.block_infos[state.pc_initial] = adapt_state_infos(state)

        _logger.debug('block_infos: %s', self.block_infos.keys())
        assert self.block_infos

    def _is_implicit_new_block(self, state):
        inst = state.get_inst()

        if inst.offset in self._bytecode.labels:
            return True
        elif inst.opname in NEW_BLOCKERS:
            return True
        else:
            return False


class Runner(object):
    def __init__(self):
        self.pending = []
        self.finished = set()

    def dispatch(self, state):
        inst = state.get_inst()
        _logger.debug("dispatch pc=%s, inst=%s", state._pc, inst)
        fn = getattr(self, 'op_{}'.format(inst.opname))
        fn(state, inst)

    def op_POP_TOP(self, state, inst):
        state.pop()

    def op_LOAD_GLOBAL(self, state, inst):
        res = state.make_temp()
        state.append(inst, res=res)
        state.push(res)

    def op_LOAD_CONST(self, state, inst):
        res = state.make_temp('const')
        state.push(res)
        state.append(inst, res=res)

    def op_LOAD_ATTR(self, state, inst):
        item = state.pop()
        res = state.make_temp()
        state.append(inst, item=item, res=res)
        state.push(res)

    def op_LOAD_FAST(self, state, inst):
        name = state.get_varname(inst)
        res = state.make_temp(name)
        state.append(inst, res=res)
        state.push(res)

    def op_STORE_FAST(self, state, inst):
        value = state.pop()
        state.append(inst, value=value)

    def op_SLICE_1(self, state, inst):
        """
        TOS = TOS1[TOS:]
        """
        tos = state.pop()
        tos1 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, start=tos, res=res, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)
        state.push(res)

    def op_SLICE_2(self, state, inst):
        """
        TOS = TOS1[:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, stop=tos, res=res, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)
        state.push(res)

    def op_SLICE_3(self, state, inst):
        """
        TOS = TOS2[TOS1:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(inst, base=tos2, start=tos1, stop=tos, res=res,
                    slicevar=slicevar, indexvar=indexvar)
        state.push(res)

    def op_STORE_SLICE_0(self, state, inst):
        """
        TOS[:] = TOS1
        """
        tos = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos, value=value, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_1(self, state, inst):
        """
        TOS1[TOS:] = TOS2
        """
        tos = state.pop()
        tos1 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, start=tos, slicevar=slicevar,
                    value=value, indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_2(self, state, inst):
        """
        TOS1[:TOS] = TOS2
        """
        tos = state.pop()
        tos1 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, stop=tos, value=value, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_3(self, state, inst):
        """
        TOS2[TOS1:TOS] = TOS3
        """
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(inst, base=tos2, start=tos1, stop=tos, value=value,
                    slicevar=slicevar, indexvar=indexvar)

    def op_DELETE_SLICE_0(self, state, inst):
        """
        del TOS[:]
        """
        tos = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_1(self, state, inst):
        """
        del TOS1[TOS:]
        """
        tos = state.pop()
        tos1 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, start=tos, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_2(self, state, inst):
        """
        del TOS1[:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, stop=tos, slicevar=slicevar,
                    indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_3(self, state, inst):
        """
        del TOS2[TOS1:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(inst, base=tos2, start=tos1, stop=tos,
                    slicevar=slicevar, indexvar=indexvar)

    def op_BUILD_SLICE(self, state, inst):
        """
        slice(TOS1, TOS) or slice(TOS2, TOS1, TOS)
        """
        argc = inst.arg
        if argc == 2:
            tos = state.pop()
            tos1 = state.pop()
            start = tos1
            stop = tos
            step = None
        elif argc == 3:
            tos = state.pop()
            tos1 = state.pop()
            tos2 = state.pop()
            start = tos2
            stop = tos1
            step = tos
        else:
            raise Exception("unreachable")
        slicevar = state.make_temp()
        res = state.make_temp()
        state.append(inst, start=start, stop=stop, step=step, res=res,
                    slicevar=slicevar)
        state.push(res)

    def _op_POP_JUMP_IF(self, state, inst):
        pred = state.pop()
        state.append(inst, pred=pred)
        state.fork(pc=inst.next)
        state.fork(pc=inst.get_jump_target())

    op_POP_JUMP_IF_TRUE = _op_POP_JUMP_IF
    op_POP_JUMP_IF_FALSE = _op_POP_JUMP_IF

    def _op_JUMP_IF_OR_POP(self, state, inst):
        pred = state.get_tos()
        state.append(inst, pred=pred)
        state.fork(pc=inst.next, npop=1)
        state.fork(pc=inst.get_jump_target())

    op_JUMP_IF_FALSE_OR_POP = _op_JUMP_IF_OR_POP
    op_JUMP_IF_TRUE_OR_POP = _op_JUMP_IF_OR_POP


    def op_JUMP_ABSOLUTE(self, state, inst):
        state.append(inst)
        state.fork(pc=inst.get_jump_target())

    def op_RETURN_VALUE(self, state, inst):
        state.append(inst, retval=state.pop(), castval=state.make_temp())
        state.terminate()

    def op_BINARY_SUBSCR(self, state, inst):
        index = state.pop()
        target = state.pop()
        res = state.make_temp()
        state.append(inst, index=index, target=target, res=res)
        state.push(res)

    def op_STORE_SUBSCR(self, state, inst):
        index = state.pop()
        target = state.pop()
        value = state.pop()
        state.append(inst, target=target, index=index, value=value)

    def op_CALL_FUNCTION(self, state, inst):
        narg = inst.arg
        args = list(reversed([state.pop() for _ in range(narg)]))
        func = state.pop()

        res = state.make_temp()
        state.append(inst, func=func, args=args, res=res)
        state.push(res)

    def op_UNPACK_SEQUENCE(self, state, inst):
        count = inst.arg
        iterable = state.pop()
        stores = [state.make_temp() for _ in range(count)]
        tupleobj = state.make_temp()
        state.append(inst, iterable=iterable, stores=stores, tupleobj=tupleobj)
        for st in reversed(stores):
            state.push(st)

    def op_BUILD_TUPLE(self, state, inst):
        count = inst.arg
        items = list(reversed([state.pop() for _ in range(count)]))
        tup = state.make_temp()
        state.append(inst, items=items, res=tup)
        state.push(tup)

    def op_GET_ITER(self, state, inst):
        value = state.pop()
        res = state.make_temp()
        state.append(inst, value=value, res=res)
        state.push(res)

    def op_FOR_ITER(self, state, inst):
        iterator = state.get_tos()
        pair = state.make_temp()
        indval = state.make_temp()
        pred = state.make_temp()
        state.append(inst, iterator=iterator, pair=pair, indval=indval, pred=pred)
        state.push(indval)
        end = inst.get_jump_target()
        state.fork(pc=end, npop=2)
        state.fork(pc=inst.next)

    def _binaryop(self, stack, inst):
        rhs = stack.pop()
        lhs = stack.pop()
        res = stack.make_temp()
        stack.append(inst, lhs=lhs, rhs=rhs, res=res)
        stack.push(res)

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

    #NOTE: Please see notes in `interpreter.py` surrounding the implementation
    # of LOAD_METHOD and CALL_METHOD.

    def op_LOAD_METHOD(self, state, inst):
        self.op_LOAD_ATTR(state, inst)

    def op_CALL_METHOD(self, state, inst):
        self.op_CALL_FUNCTION(state, inst)



class State(object):
    def __init__(self, bytecode, pc, nstack):
        self._bytecode = bytecode
        self._pc_initial = pc
        self._pc = pc
        self._nstack_initial = nstack
        self._stack = []
        self._blockstack = []
        self._temp_registers = []
        self._insts = []
        self._outedges = []
        self._terminated = False
        self._phis = {}
        self._outgoing_phis = UniqueDict()
        for i in range(nstack):
            phi = self.make_temp('phi')
            self._phis[phi] = i
            self.push(phi)

    def __repr__(self):
        return "State(pc_initial={} nstack_initial={})".format(
            self._pc_initial, self._nstack_initial,
        )

    def get_identity(self):
        return (
            self._pc_initial,
            self._nstack_initial,
        )

    def __hash__(self):
        return hash(self.get_identity())

    def __eq__(self, other):
        return self.get_identity() == other.get_identity()

    @property
    def pc_initial(self):
        return self._pc_initial

    @property
    def instructions(self):
        return self._insts

    @property
    def outgoing_edges(self):
        return self._outedges

    @property
    def outgoing_phis(self):
        return self._outgoing_phis

    def has_terminated(self):
        return self._terminated

    def get_inst(self):
        return self._bytecode[self._pc]

    def advance_pc(self):
        inst = self.get_inst()
        self._pc = inst.next

    def make_temp(self, prefix=''):
        name = '${prefix}{offset}{opname}.{tempct}'.format(
            prefix=prefix,
            offset=self._pc,
            opname=self.get_inst().opname,
            tempct=len(self._temp_registers),
        )
        self._temp_registers.append(name)
        return name

    def append(self, inst, **kwargs):
        """Append new inst"""
        self._insts.append((inst.offset, kwargs))

    def get_tos(self):
        item = self.pop()
        self.push(item)
        return item

    def push(self, item):
        """Push to stack"""
        self._stack.append(item)

    def pop(self):
        """Pop the stack"""
        return self._stack.pop()

    def get_varname(self, inst):
        """Get referenced variable name from the oparg
        """
        return self._bytecode.co_varnames[inst.arg]

    def terminate(self):
        """Mark block as terminated
        """
        self._terminated = True

    def fork(self, pc, npop=0):
        """Fork the state
        """
        assert 0 <= npop <= len(self._stack)
        nstack = len(self._stack) - npop
        stack = self._stack[:nstack]
        self._outedges.append(Edge(pc=pc, stack=stack))
        self.terminate()

    def split_new_block(self):
        """Split the state
        """
        self.fork(pc=self._pc)

    def get_outgoing_states(self):
        """Get states for each outgoing edges
        """
        # Should only call once
        assert not self._outgoing_phis
        ret = []
        for edge in self._outedges:
            state = State(bytecode=self._bytecode, pc=edge.pc,
                          nstack=len(edge.stack))
            ret.append(state)
            # Map outgoing_phis
            for phi, i in state._phis.items():
                self._outgoing_phis[phi] = edge.stack[i]
        return ret


Edge = namedtuple('Edge', ['pc', 'stack'])



class AdaptDFA(object):
    def __init__(self, flow):
        self._flow = flow

    @property
    def infos(self):
        return self._flow.block_infos


AdaptBlockInfo = namedtuple('AdaptBlockInfo', [
    'insts',
    'outgoing_phis',
])


def adapt_state_infos(state):
    return AdaptBlockInfo(
        insts=tuple(state.instructions),
        outgoing_phis=state.outgoing_phis,
    )