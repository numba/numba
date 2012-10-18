#! /usr/bin/env python
# ______________________________________________________________________

from bytecode_visitor import BytecodeFlowVisitor, BenignBytecodeVisitorMixin

# ______________________________________________________________________

synthetic_opname = []
synthetic_opmap = {}

def def_synth_op (opname):
    global synthetic_opname, synthetic_opmap
    ret_val = -(len(synthetic_opname) + 1)
    synthetic_opname.insert(0, opname)
    synthetic_opmap[opname] = ret_val
    return ret_val

REF_ARG = def_synth_op('REF_ARG')
BUILD_PHI = def_synth_op('BUILD_PHI')
DEFINITION = def_synth_op('DEFINITION')
REF_DEF = def_synth_op('REF_DEF')

# ______________________________________________________________________

class PhiInjector (BenignBytecodeVisitorMixin, BytecodeFlowVisitor):
    '''Transformer responsible for modifying a bytecode flow, removing
    LOAD_FAST and STORE_FAST opcodes, and replacing them with a static
    single assignment (SSA) representation.

    In order to support SSA, PhiInjector adds the following synthetic
    opcodes to transformed flows:

      * REF_ARG: Specifically reference an incomming argument value.

      * BUILD_PHI: Build a phi node to disambiguate between several
        possible definitions at a control flow join.

      * DEFINITION: Unique value definition indexed by the "arg" field
        in the tuple.

      * REF_DEF: Reference a specific value definition.'''

    def visit_cfg (self, cfg, nargs = 0, *args, **kws):
        self.cfg = cfg
        ret_val = self.visit(cfg.blocks, nargs)
        del self.cfg
        return ret_val

    def visit (self, flow, nargs = 0, *args, **kws):
        self.nargs = nargs
        self.definitions = []
        self.phis = []
        self.prev_blocks = []
        self.blocks_locals = dict((block, {})
                                  for block in self.cfg.blocks.keys())
        ret_val = super(PhiInjector, self).visit(flow, *args, **kws)
        for block, _, _, args, _ in self.phis:
            local = args.pop()
            reaching_definitions = self.cfg.reaching_definitions[block]
            for prev in reaching_definitions.keys():
                if 0 in self.cfg.blocks_reaching[prev]:
                    args.append((prev, REF_DEF, 'REF_DEF',
                                 self.blocks_locals[prev][local], ()))
            args.sort()
        del self.blocks_locals
        del self.prev_blocks
        del self.phis
        del self.definitions
        del self.nargs
        return ret_val

    def add_definition (self, index, local, arg):
        definition_index = len(self.definitions)
        definition = (index, DEFINITION, 'DEFINITION', definition_index,
                      (arg,))
        self.definitions.append(definition)
        self.blocks_locals[self.block][local] = definition_index
        return definition

    def add_phi (self, index, local):
        ret_val = (index, BUILD_PHI, 'BUILD_PHI', [local], ())
        self.phis.append(ret_val)
        return ret_val

    def enter_block (self, block):
        ret_val = False
        self.block = block
        if block == 0:
            if self.nargs > 0:
                ret_val = [self.add_definition(-1, arg,
                                                (-1, REF_ARG, 'REF_ARG', arg,
                                                  ()))
                           for arg in range(self.nargs)]
            else:
                ret_val = True
        elif 0 in self.cfg.blocks_reaching[block]:
            ret_val = True
            prev_block_locals = None
            for pred_block in self.cfg.blocks_in[block]:
                if pred_block in self.prev_blocks:
                    prev_block_locals = self.blocks_locals[pred_block]
                    break
            assert prev_block_locals is not None, "Internal translation error"
            self.blocks_locals[block] = prev_block_locals.copy()
            phis_needed = self.cfg.phi_needed(block)
            if phis_needed:
                ret_val = [self.add_definition(block, local,
                                               self.add_phi(block, local))
                           for local in phis_needed]
        return ret_val

    def exit_block (self, block):
        if 0 in self.cfg.blocks_reaching[block]:
            self.prev_blocks.append(block)
        del self.block

    def op_STORE_FAST (self, i, op, arg, *args, **kws):
        assert len(args) == 1
        return [self.add_definition(i, arg, args[0])]

    def op_LOAD_FAST (self, i, op, arg, *args, **kws):
        return [(i, REF_DEF, 'REF_DEF', self.blocks_locals[self.block][arg],
                 args)]

# ______________________________________________________________________

def inject_phis (func):
    '''Given a Python function, return a bytecode flow object that has
    been transformed by a fresh PhiInjector instance.'''
    import byte_control
    argcount = byte_control.opcode_util.get_code_object(func).co_argcount
    cfg = byte_control.build_cfg(func)
    return PhiInjector().visit_cfg(cfg, argcount)

# ______________________________________________________________________
# Main (self-test) routine

def main (*args):
    import pprint
    from tests import llfuncs
    if not args:
        args = ('doslice',)
    for arg in args:
        pprint.pprint(inject_phis(getattr(llfuncs, arg)))

# ______________________________________________________________________

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of phi_injector.py
