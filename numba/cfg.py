#! /usr/bin/env python
# ______________________________________________________________________

import opcode

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

    @classmethod
    def build_cfg (cls, code_obj, *args, **kws):
        from .translate import itercode
        ret_val = cls(*args, **kws)
        opmap = opcode.opmap
        JUMP_ABSOLUTE = opmap['JUMP_ABSOLUTE']
        JUMP_FORWARD = opmap['JUMP_FORWARD']
        JUMP_IF_FALSE_OR_POP = opmap['JUMP_IF_FALSE_OR_POP']
        JUMP_IF_TRUE_OR_POP = opmap['JUMP_IF_TRUE_OR_POP']
        LOAD_FAST = opmap['LOAD_FAST']
        POP_JUMP_IF_FALSE = opmap['POP_JUMP_IF_FALSE']
        POP_JUMP_IF_TRUE = opmap['POP_JUMP_IF_TRUE']
        RETURN_VALUE = opmap['RETURN_VALUE']
        SETUP_LOOP = opmap['SETUP_LOOP']
        STORE_FAST = opmap['STORE_FAST']
        crnt_block = 0
        ret_val.add_block(0)
        ret_val.blocks_writes[0] = set(range(code_obj.co_argcount))
        last_was_jump = True # At start there is no prior basic block
                             # to link up with, so skip building a
                             # fallthrough edge.
        for i, op, arg in itercode(code_obj.co_code):
            if i in ret_val.blocks:
                if not last_was_jump:
                    ret_val.add_edge(crnt_block, i)
                crnt_block = i
            last_was_jump = False
            if op == JUMP_ABSOLUTE:
                ret_val.add_block(arg)
                ret_val.add_edge(crnt_block, arg)
                ret_val.add_block(i + 3)
                last_was_jump = True
            elif op == JUMP_FORWARD:
                target = i + arg + 3
                ret_val.add_block(target)
                ret_val.add_edge(crnt_block, target)
                ret_val.add_block(i + 3)
                last_was_jump = True
            elif op in (JUMP_IF_FALSE_OR_POP, JUMP_IF_TRUE_OR_POP):
                raise NotImplementedError('FIXME')
            elif op == LOAD_FAST:
                # Not sure if we care about local variable users at
                # the moment (more concerned with reaches analysis).
                # Might help us eliminate unneeded phi nodes...
                ret_val.blocks_reads[crnt_block].add(arg)
            elif op in (POP_JUMP_IF_FALSE, POP_JUMP_IF_TRUE):
                ret_val.add_block(i + 3)
                ret_val.add_block(arg)
                ret_val.add_edge(crnt_block, i + 3)
                ret_val.add_edge(crnt_block, arg)
                last_was_jump = True
            elif op == RETURN_VALUE:
                if i + 1 < len(code_obj.co_code):
                    ret_val.add_block(i + 1)
                last_was_jump = True
            elif op == SETUP_LOOP:
                ret_val.add_block(i + 3)
                ret_val.add_edge(crnt_block, i + 3)
                last_was_jump = True # Not technically true, but we've
                                     # already built the proper CFG
                                     # edges, so skip the fallthrough
                                     # plumbing.
            elif op == STORE_FAST:
                ret_val.blocks_writes[crnt_block].add(arg)
                ret_val.blocks_writer[crnt_block][arg] = i
        return ret_val

    def compute_dom (self):
        '''Compute the dominator relationship in the CFG.'''
        for block in self.blocks.iterkeys():
            self.blocks_dom[block] = set((block,))
        changed = True
        while changed:
            changed = False
            for block in self.blocks.keys():
                olddom = self.blocks_dom[block]
                newdom = olddom.union(*[self.blocks_dom[pred]
                                        for pred in self.blocks_in[block]])
                if newdom != olddom:
                    changed = True
                    self.blocks_dom[block] = newdom
        return self.blocks_dom

    def idom (self, block):
        '''Compute the immediate dominator (idom) of the given block
        key.  Returns None if the block has no in edges.

        Note that in the case where there are multiple immediate
        dominators (a join after a non-loop branch), this returns one
        of the predecessors, but is not guaranteed to reliably select
        one over the others (depends on the order used by iterators
        over sets).

        Since our data structure stores back edges, we can skip the
        naive, O(n^2), approach to finding the idom of a given block.'''
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
        if hasattr(self, 'reaching_defns'):
            ret_val = self.reaching_defns
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
            self.reaching_defns = ret_val
        return ret_val

    def nreaches (self, block):
        '''For each local, find the number of unique reaching
        definitions the current block has.'''
        # Slice and dice the idom tree so that each predecessor claims
        # at most one definition so we don't end up over or
        # undercounting.
        preds = self.blocks_in[block]
        idoms = {}
        idom_writes = {}
        # Fib a little here to truncate traversal in loops if they are
        # being chased before the actual idom of the current block has
        # been handled.
        visited = preds.copy()
        for pred in preds:
            idoms[pred] = set((pred,))
            idom_writes[pred] = self.blocks_writes[pred].copy()
            # Traverse up the idom tree, adding sets of writes as we
            # go.
            crnt = self.idom(pred)
            while crnt != None and crnt not in visited:
                idoms[pred].add(crnt)
                idom_writes[pred].update(self.blocks_writes[crnt])
                visited.add(crnt)
                crnt = self.idom(crnt)
        all_writes = set.union(*[idom_writes[pred] for pred in preds])
        ret_val = {}
        for local in all_writes:
            ret_val[local] = sum((1 if local in idom_writes[pred] else 0
                                  for pred in preds))
        return ret_val

    def phi_needed (self, join):
        '''Return the set of locals that will require a phi node to be
        generated at the given join.'''
        nreaches = self.nreaches(join)
        return set([local for local in nreaches.iterkeys()
                    if nreaches[local] > 1])

# ______________________________________________________________________
# End of cfg.py
