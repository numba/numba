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
        ret_val = cls(*args, **kws)
        opmap = opcode.opname
        ret_val.crnt_block = 0
        ret_val.code_len = len(code_obj.co_code)
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
        del ret_val.crnt_block, ret_val.code_len
        return ret_val

    # NOTE: The following op_OPNAME methods are correct for Python
    # semantics, but may be overloaded for Numba-specific semantics.

    def op_FOR_ITER (self, i, op, arg):
        self.add_block(i)
        self.add_edge(self.crnt_block, i)
        self.add_block(i + arg + 3)
        self.add_edge(i, i + arg + 3)
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
        return True # This is not technically a jump, but we've
                    # already built the proper CFG edges, so skip the
                    # fallthrough plumbing.

    def op_STORE_FAST (self, i, op, arg):
        self.blocks_writes[self.crnt_block].add(arg)
        self.blocks_writer[self.crnt_block][arg] = i
        return False

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

    def pprint (self, *args, **kws):
        pprint.pprint((self.blocks_in, self.blocks_out, self.blocks_reads,
                       self.blocks_writes, self.blocks_dom), *args, **kws)

# ______________________________________________________________________

def main (*args, **kws):
    import importlib
    for arg in args:
        module_split = arg.rsplit('.', 1)
        if len(module_split) != 2:
            print("Don't know how to handle %r, expecting <module.member> "
                  "arguments.  Skipping..." % (arg,))
        else:
            module = importlib.import_module(module_split[0])
            func = getattr(module, module_split[1])
            if not hasattr(func, 'func_code'):
                print("Don't know how to handle %r, module member does not "
                      "have a code object.  Skipping..." % (arg,))
            else:
                cfg = ControlFlowGraph.build_cfg(func.func_code)
                cfg.compute_dom()
                cfg.pprint()

# ______________________________________________________________________

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of cfg.py
