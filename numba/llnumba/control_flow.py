#! /usr/bin/env python
# ______________________________________________________________________

import pprint

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
        joins = [block for block in self.blocks.keys()
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
                        self.writes_local(block, block, affected_local)
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
            for local in reaching_from_pred.keys():
                if local not in definition_map:
                    definition_map[local] = set()
                definition_map[local].add(reaching_from_pred[local])
        ret_val = {}
        for local in definition_map.keys():
            ret_val[local] = len(definition_map[local])
        return ret_val

    def writes_local (self, block, write_instr_index, local_index):
        self.blocks_writes[block].add(local_index)
        block_writers = self.blocks_writer[block]
        old_index = block_writers.get(local_index, -1)
        # This checks for a corner case that would impact
        # numba.translate.Translate.build_phi_nodes().
        assert old_index != write_instr_index, (
            "Found corner case for STORE_FAST at a CFG join!")
        block_writers[local_index] = max(write_instr_index, old_index)

    def phi_needed (self, join):
        '''Return the set of locals that will require a phi node to be
        generated at the given join.'''
        nreaches = self.nreaches(join)
        return set([local for local in nreaches.keys()
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
# End of control_flow.py
