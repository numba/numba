from __future__ import print_function, division, absolute_import
import functools
from numba import utils


NEW_BLOCKERS = frozenset(['SETUP_LOOP'])


class CFBlock(object):
    def __init__(self, offset):
        self.offset = offset
        self.body = []
        self.outgoing = set()
        self.incoming = set()
        self.terminating = False

    def __repr__(self):
        args = self.body, self.outgoing, self.incoming
        return "block(body: %s, outgoing: %s, incoming: %s)" % args

    def __iter__(self):
        return iter(self.body)


class ControlFlowAnalysis(object):
    """
    Attributes
    ----------
    - bytecode

    - blocks

    - blockseq

    - doms: dict of set
        Dominators

    - backbone: set of block offsets
        The set of block that is common to all possible code path.

    """
    def __init__(self, bytecode):
        self.bytecode = bytecode
        self.blocks = {}
        self.liveblocks = {}
        self.blockseq = []
        self.doms = None
        self.backbone = None
        # Internal temp states
        self._force_new_block = True
        self._curblock = None
        self._blockstack = []
        self._loops = []

    def iterblocks(self):
        """
        Return all blocks in sequence of occurrence
        """
        for i in self.blockseq:
            yield self.blocks[i]

    def iterliveblocks(self):
        """
        Return all live blocks in sequence of occurrence
        """
        for i in self.blockseq:
            if i in self.liveblocks:
                yield self.blocks[i]

    def run(self):
        for inst in self._iter_inst():
            fname = "op_%s" % inst.opname
            fn = getattr(self, fname, None)
            if fn is not None:
                fn(inst)
            else:
                assert not inst.is_jump, inst

        # Close all blocks
        for cur, nxt in zip(self.blockseq, self.blockseq[1:]):
            blk = self.blocks[cur]
            if not blk.outgoing and not blk.terminating:
                blk.outgoing.add(nxt)

        # Fill incoming
        for b in utils.dict_itervalues(self.blocks):
            for out in b.outgoing:
                self.blocks[out].incoming.add(b.offset)

        # Find liveblocks
        self.dead_block_elimin()

        # Find dominators
        self.doms = find_dominators(self.liveblocks)

        for lastblk in reversed(self.blockseq):
            if lastblk in self.liveblocks:
                break
        else:
            raise AssertionError("No live block that exits!?")

        # Find backbone
        backbone = set(self.doms[lastblk])
        # Filter out in loop blocks (Assuming no other cyclic control blocks)
        inloopblocks = set()
        for b in self.blocks.keys():
            for s, e in self._loops:
                if s <= b < e:
                    inloopblocks.add(b)
        self.backbone = backbone - inloopblocks

    def dead_block_elimin(self):
        liveset = set([0])
        pending = set([0])
        finished = set()
        while pending:
            cur = pending.pop()
            blk = self.blocks[cur]
            outgoing = set(blk.outgoing)
            liveset |= outgoing
            pending |= outgoing - finished
            finished.add(cur)

        for offset in liveset:
            self.liveblocks[offset] = self.blocks[offset]

    def jump(self, target):
        self._curblock.outgoing.add(target)

    def _iter_inst(self):
        for inst in self.bytecode:
            if self._use_new_block(inst):
                self._start_new_block(inst)
            self._curblock.body.append(inst.offset)
            yield inst

    def _use_new_block(self, inst):
        if inst.offset in self.bytecode.labels:
            res = True
        elif inst.opname in NEW_BLOCKERS:
            res = True
        else:
            res = self._force_new_block

        self._force_new_block = False
        return res

    def _start_new_block(self, inst):
        self._curblock = CFBlock(inst.offset)
        self.blocks[inst.offset] = self._curblock
        self.blockseq.append(inst.offset)

    def op_SETUP_LOOP(self, inst):
        end = inst.get_jump_target()
        self._blockstack.append(end)
        self._loops.append((inst.offset, end))

    def op_POP_BLOCK(self, inst):
        self._blockstack.pop()

    def op_FOR_ITER(self, inst):
        self.jump(inst.get_jump_target())
        self.jump(inst.next)
        self._force_new_block = True

    def _op_ABSOLUTE_JUMP_IF(self, inst):
        self.jump(inst.get_jump_target())
        self.jump(inst.next)
        self._force_new_block = True

    op_POP_JUMP_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_IF_TRUE = _op_ABSOLUTE_JUMP_IF
    op_JUMP_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_JUMP_IF_TRUE = _op_ABSOLUTE_JUMP_IF
    op_JUMP_IF_FALSE_OR_POP = _op_ABSOLUTE_JUMP_IF
    op_JUMP_IF_TRUE_OR_POP = _op_ABSOLUTE_JUMP_IF

    def op_JUMP_ABSOLUTE(self, inst):
        self.jump(inst.get_jump_target())
        self._force_new_block = True

    def op_JUMP_FORWARD(self, inst):
        self.jump(inst.get_jump_target())
        self._force_new_block = True

    def op_RETURN_VALUE(self, inst):
        self._curblock.terminating = True
        self._force_new_block = True

    def op_BREAK_LOOP(self, inst):
        self.jump(self._blockstack[-1])
        self._force_new_block = True


def find_dominators(blocks):
    doms = {}
    for b in blocks:
        doms[b] = set()

    doms[0].add(0)
    allblks = set(blocks)

    remainblks = frozenset(blk.offset
                           for blk in utils.dict_values(blocks)
                           if blk.offset != 0)
    for blk in remainblks:
        doms[blk] |= allblks

    changed = True
    while changed:
        changed = False
        for blk in remainblks:
            d = doms[blk]
            ps = [doms[p] for p in blocks[blk].incoming if p in doms]
            if not ps:
                p = set()
            else:
                p = functools.reduce(set.intersection, ps)
            new = set([blk]) | p
            if new != d:
                doms[blk] = new
                changed = True

    return doms
