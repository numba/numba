#! /usr/bin/env python
# ______________________________________________________________________
'''cfg

Defines the ControlFlowGraph class, which is used by the Numba
translator to perform accurate phi node generation.

When used as the main module, displays the control flow graph for
arguments of the form <module.module_fn>.  Example:

% python -m numba.cfg test_while.while_loop_fn_0
'''
# ______________________________________________________________________

import opcode
import pprint

from .utils import itercode

import sys

import logging

logger = logging.getLogger(__name__)

# ______________________________________________________________________

class ControlFlowGraph (object):
    def __init__ (self):
        self.blocks = {}
        self.blocks_in = {}
        self.blocks_out = {}
        self.blocks_reads = {}
        self.blocks_writes = {}
        self.blocks_writer = {}
        self.blocks_dom = {}
        self.blocks_reaching = {}

    def add_block (self, key, value = None):
        self.blocks[key] = value
        if key not in self.blocks_in:
            self.blocks_in[key] = set()
            self.blocks_out[key] = set()
            self.blocks_reads[key] = set()
            self.blocks_writes[key] = set()
            self.blocks_writer[key] = {}

    def add_edge (self, from_block, to_block):
        self.blocks_out[from_block].add(to_block)
        self.blocks_in[to_block].add(from_block)

    def unlink_unreachables (self):
        changed = True
        next_blocks = self.blocks.keys()
        next_blocks.remove(0)
        while changed:
            changed = False
            blocks = next_blocks
            next_blocks = blocks[:]
            for block in blocks:
                if len(self.blocks_in[block]) == 0:
                    blocks_out = self.blocks_out[block]
                    for out_edge in blocks_out:
                        self.blocks_in[out_edge].discard(block)
                    blocks_out.clear()
                    next_blocks.remove(block)
                    changed = True

    @classmethod
    def build_cfg (cls, code_obj, *args, **kws):
        ret_val = cls(*args, **kws)
        opmap = opcode.opname
        ret_val.crnt_block = 0
        ret_val.code_len = len(code_obj.co_code)
        ret_val.loop_stack = []
        ret_val.add_block(0)
        ret_val.blocks_writes[0] = set(range(code_obj.co_argcount))
        last_was_jump = True # At start there is no prior basic block
                             # to link up with, so skip building a
                             # fallthrough edge.
        for i, op, arg in itercode(code_obj.co_code):
            if i in ret_val.blocks:
                if not last_was_jump:
                    ret_val.add_edge(ret_val.crnt_block, i)
                ret_val.crnt_block = i
            last_was_jump = False
            method_name = "op_" + opmap[op]
            if hasattr(ret_val, method_name):
                last_was_jump = getattr(ret_val, method_name)(i, op, arg)
        ret_val.unlink_unreachables()
        del ret_val.crnt_block, ret_val.code_len, ret_val.loop_stack
        return ret_val

    # NOTE: The following op_OPNAME methods are correct for Python
    # semantics, but may be overloaded for Numba-specific semantics.

    def op_FOR_ITER (self, i, op, arg):
        self.add_block(i)
        self.add_edge(self.crnt_block, i)
        self.add_block(i + arg + 3)
        self.add_edge(i, i + arg + 3)
        self.add_block(i + 3)
        self.add_edge(i, i + 3)
        self.crnt_block = i
        return False

    def op_JUMP_ABSOLUTE (self, i, op, arg):
        self.add_block(arg)
        self.add_edge(self.crnt_block, arg)
        self.add_block(i + 3)
        return True

    def op_JUMP_FORWARD (self, i, op, arg):
        target = i + arg + 3
        self.add_block(target)
        self.add_edge(self.crnt_block, target)
        self.add_block(i + 3)
        return True

    def op_JUMP_IF_FALSE_OR_POP (self, i, op, arg):
        raise NotImplementedError('FIXME')

    op_JUMP_IF_TRUE_OR_POP = op_JUMP_IF_FALSE_OR_POP

    def op_LOAD_FAST (self, i, op, arg):
        self.blocks_reads[self.crnt_block].add(arg)
        return False

    def op_POP_JUMP_IF_FALSE (self, i, op, arg):
        self.add_block(i + 3)
        self.add_block(arg)
        self.add_edge(self.crnt_block, i + 3)
        self.add_edge(self.crnt_block, arg)
        return True

    op_POP_JUMP_IF_TRUE = op_POP_JUMP_IF_FALSE

    def op_RETURN_VALUE (self, i, op, arg):
        if i + 1 < self.code_len:
            self.add_block(i + 1)
        return True

    def op_SETUP_LOOP (self, i, op, arg):
        self.add_block(i + 3)
        self.add_edge(self.crnt_block, i + 3)
        self.loop_stack.append((i, arg))
        return True # This is not technically a jump, but we've
                    # already built the proper CFG edges, so skip the
                    # fallthrough plumbing.

    def op_BREAK_LOOP (self, i, op, arg):
        loop_i, loop_arg = self.loop_stack[-1]
        loop_exit = loop_i + 3 + loop_arg
        self.add_block(loop_exit)
        self.add_edge(self.crnt_block, loop_exit)
        self.add_block(i + 1)
        return True

    def op_POP_BLOCK (self, i, op, arg):
        self.loop_stack.pop()
        return False

    def _writes_local (self, block, write_instr_index, local_index):
        self.blocks_writes[block].add(local_index)
        block_writers = self.blocks_writer[block]
        old_index = block_writers.get(local_index, -1)
        # This checks for a corner case that would impact
        # numba.translate.Translate.build_phi_nodes().
        assert old_index != write_instr_index, (
            "Found corner case for STORE_FAST at a CFG join!")
        block_writers[local_index] = max(write_instr_index, old_index)

    def op_STORE_FAST (self, i, op, arg):
        self._writes_local(self.crnt_block, i, arg)
        return False

    def compute_dataflow (self):
        '''Compute the dominator and reaching dataflow relationships
        in the CFG.'''
        blocks = set(self.blocks.keys())
        nonentry_blocks = blocks.copy()
        for block in blocks:
            self.blocks_dom[block] = blocks
            self.blocks_reaching[block] = set((block,))
            if len(self.blocks_in[block]) == 0:
                self.blocks_dom[block] = set((block,))
                nonentry_blocks.remove(block)
        changed = True
        while changed:
            changed = False
            for block in nonentry_blocks:
                olddom = self.blocks_dom[block]
                newdom = set.intersection(*[self.blocks_dom[pred]
                                            for pred in self.blocks_in[block]])
                newdom.add(block)
                if newdom != olddom:
                    changed = True
                    self.blocks_dom[block] = newdom
                oldreaching = self.blocks_reaching[block]
                newreaching = set.union(
                    *[self.blocks_reaching[pred]
                      for pred in self.blocks_in[block]])
                newreaching.add(block)
                if newreaching != oldreaching:
                    changed = True
                    self.blocks_reaching[block] = newreaching
        return self.blocks_dom, self.blocks_reaching

    def update_for_ssa (self):
        '''Modify the blocks_writes map to reflect phi nodes inserted
        for static single assignment representations.'''
        joins = [block for block in self.blocks.iterkeys()
                 if len(self.blocks_in[block]) > 1]
        changed = True
        while changed:
            changed = False
            for block in joins:
                phis_needed = self.phi_needed(block)
                for affected_local in phis_needed:
                    if affected_local not in self.blocks_writes[block]:
                        changed = True
                        # NOTE: For this to work, we assume that basic
                        # blocks are indexed by their instruction
                        # index in the VM bytecode.
                        self._writes_local(block, block, affected_local)
            if changed:
                # Any modifications have invalidated the reaching
                # definitions, so delete any memoized results.
                if hasattr(self, 'reaching_definitions'):
                    del self.reaching_definitions

    def idom (self, block):
        '''Compute the immediate dominator (idom) of the given block
        key.  Returns None if the block has no in edges.

        Note that in the case where there are multiple immediate
        dominators (a join after a non-loop branch), this returns one
        of the predecessors, but is not guaranteed to reliably select
        one over the others (depends on the ordering of the set type
        iterator).'''
        preds = self.blocks_in[block]
        npreds = len(preds)
        if npreds == 0:
            ret_val = None
        elif npreds == 1:
            ret_val = tuple(preds)[0]
        else:
            ret_val = [pred for pred in preds
                       if block not in self.blocks_dom[pred]][0]
        return ret_val

    def block_writes_to_writer_map (self, block):
        ret_val = {}
        for local in self.blocks_writes[block]:
            ret_val[local] = block
        return ret_val

    def get_reaching_definitions (self, block):
        '''Return a nested map for the given block
        s.t. ret_val[pred][local] equals the block key for the
        definition of local that reaches the argument block via that
        predecessor.

        Useful for actually populating phi nodes, once you know you
        need them.'''
        has_memoized = hasattr(self, 'reaching_definitions')
        if has_memoized and block in self.reaching_definitions:
            ret_val = self.reaching_definitions[block]
        else:
            preds = self.blocks_in[block]
            ret_val = {}
            for pred in preds:
                ret_val[pred] = self.block_writes_to_writer_map(pred)
                crnt = self.idom(pred)
                while crnt != None:
                    crnt_writer_map = self.block_writes_to_writer_map(crnt)
                    # This order of update favors the first definitions
                    # encountered in the traversal since the traversal
                    # visits blocks in reverse execution order.
                    crnt_writer_map.update(ret_val[pred])
                    ret_val[pred] = crnt_writer_map
                    crnt = self.idom(crnt)
            if not has_memoized:
                self.reaching_definitions = {}
            self.reaching_definitions[block] = ret_val
        return ret_val

    def nreaches (self, block):
        '''For each local, find the number of unique reaching
        definitions the current block has.'''
        reaching_definitions = self.get_reaching_definitions(block)
        definition_map = {}
        for pred in self.blocks_in[block]:
            reaching_from_pred = reaching_definitions[pred]
            for local in reaching_from_pred.iterkeys():
                if local not in definition_map:
                    definition_map[local] = set()
                definition_map[local].add(reaching_from_pred[local])
        ret_val = {}
        for local in definition_map.iterkeys():
            ret_val[local] = len(definition_map[local])
        if __debug__:
            logger.debug(pprint.pformat(ret_val))
        return ret_val

    def phi_needed (self, join):
        '''Return the set of locals that will require a phi node to be
        generated at the given join.'''
        nreaches = self.nreaches(join)
        return set([local for local in nreaches.iterkeys()
                    if nreaches[local] > 1])

    def pprint (self, *args, **kws):
        pprint.pprint(self.__dict__, *args, **kws)

    def pformat (self, *args, **kws):
        return pprint.pformat(self.__dict__, *args, **kws)

    def to_dot (self, graph_name = None):
        '''Return a dot (digraph visualizer in Graphviz) graph
        description as a string.'''
        if graph_name is None:
            graph_name = 'CFG_%d' % id(self)
        lines_out = []
        for block_index in self.blocks:
            lines_out.append(
                'BLOCK_%r [shape=box, label="BLOCK_%r\\nr: %r, w: %r"];' %
                (block_index, block_index,
                 tuple(self.blocks_reads[block_index]),
                 tuple(self.blocks_writes[block_index])))
        for block_index in self.blocks:
            for out_edge in self.blocks_out[block_index]:
                lines_out.append('BLOCK_%r -> BLOCK_%r;' %
                                 (block_index, out_edge))
        return 'digraph %s {\n%s\n}\n' % (graph_name, '\n'.join(lines_out))

# ______________________________________________________________________

def main (*args, **kws):
    import getopt, importlib
    def get_module_member (member_path):
        ret_val = None
        module_split = member_path.rsplit('.', 1)
        if len(module_split) > 1:
            module = importlib.import_module(module_split[0])
            ret_val = getattr(module, module_split[1])
        return ret_val
    opts, args = getopt.getopt(args, 'dC:D:')
    kws.update(opts)
    dot_out = None
    cls = ControlFlowGraph
    for opt_key, opt_val in kws.iteritems():
        if opt_key == '-d':
            dot_out = sys.stdout
        elif opt_key in ('-D', 'dot'):
            dot_out = open(opt_val, "w")
        elif opt_key in ('-C', 'cfg_cls'):
            cls = get_module_member(opt_val)
    for arg in args:
        func = get_module_member(arg)
        if func is None:
            print("Don't know how to handle %r, expecting <module.member> "
                  "arguments.  Skipping..." % (arg,))
        elif not hasattr(func, 'func_code'):
            print("Don't know how to handle %r, module member does not "
                  "have a code object.  Skipping..." % (arg,))
        else:
            cfg = cls.build_cfg(func.func_code)
            cfg.compute_dataflow()
            if dot_out is not None:
                dot_out.write(cfg.to_dot())
            else:
                cfg.pprint()

# ______________________________________________________________________

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of cfg.py
