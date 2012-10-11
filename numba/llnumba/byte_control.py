#! /usr/bin/env python
# ______________________________________________________________________

import opcode
import opcode_util
import pprint

from bytecode_visitor import BytecodeFlowVisitor, BenignBytecodeVisitorMixin
from control_flow import ControlFlowGraph

# ______________________________________________________________________

class ControlFlowBuilder (BenignBytecodeVisitorMixin, BytecodeFlowVisitor):
    def visit (self, flow, nargs = 0, *args, **kws):
        self.nargs = nargs
        ret_val = super(ControlFlowBuilder, self).visit(flow, *args, **kws)
        del self.nargs
        return ret_val

    def enter_flow_object (self, flow):
        super(ControlFlowBuilder, self).enter_flow_object(flow)
        self.flow = flow
        self.cfg = ControlFlowGraph()
        for block in flow.keys():
            self.cfg.add_block(block, flow[block])

    def exit_flow_object (self, flow):
        super(ControlFlowBuilder, self).exit_flow_object(flow)
        assert self.flow == flow
        self.cfg.compute_dataflow()
        self.cfg.update_for_ssa()
        ret_val = self.cfg
        del self.cfg
        del self.flow
        return ret_val

    def enter_block (self, block):
        self.block = block
        assert block in self.cfg.blocks
        if block == 0:
            for local_index in range(self.nargs):
                self.op_STORE_FAST(0, opcode.opmap['STORE_FAST'], local_index)
        return True

    def _get_next_block (self, block):
        return self.block_list[self.block_list.index(block) + 1]

    def exit_block (self, block):
        assert block == self.block
        del self.block
        i, op, opname, arg, args = self.flow[block][-1]
        if op in opcode.hasjabs:
            self.cfg.add_edge(block, arg)
        elif op in opcode.hasjrel:
            self.cfg.add_edge(block, i + arg + 3)
        elif opname == 'BREAK_LOOP':
            self.cfg.add_edge(block, arg)
        elif opname != 'RETURN_VALUE':
            self.cfg.add_edge(block, self._get_next_block(block))
        if op in opcode_util.hascbranch:
            self.cfg.add_edge(block, self._get_next_block(block))

    def op_LOAD_FAST (self, i, op, arg, *args, **kws):
        self.cfg.blocks_reads[self.block].add(arg)
        return super(ControlFlowBuilder, self).op_LOAD_FAST(i, op, arg, *args,
                                                            **kws)

    def op_STORE_FAST (self, i, op, arg, *args, **kws):
        self.cfg.writes_local(self.block, i, arg)
        return super(ControlFlowBuilder, self).op_STORE_FAST(i, op, arg, *args,
                                                             **kws)

# ______________________________________________________________________

def build_cfg (func):
    import byte_flow
    return ControlFlowBuilder().visit(
        byte_flow.build_flow(func),
        opcode_util.get_code_object(func).co_argcount)

# ______________________________________________________________________
# Main (self-test) routine

def main (*args, **kws):
    from tests import llfuncs
    if not args:
        args = ('doslice',)
    for arg in args:
        build_cfg(getattr(llfuncs, arg)).pprint()

# ______________________________________________________________________

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of byte_control.py
