"""
Contains optimization passes for the IR.
"""
from __future__ import print_function, division, absolute_import
from numba import ir, utils


class RemoveRedundantAssign(object):
    """
    Turn assignment pairs into one assignment
    """
    def __init__(self, interp):
        self.interp = interp

    def run(self):
        for blkid, blk in utils.iteritems(self.interp.blocks):
            self.run_block(blk)

    def run_block(self, blk):
        tempassign = {}
        removeset = set()

        for offset, inst in enumerate(blk.body):
            self.mark_asssignment(tempassign, offset, inst)

        for bag in utils.itervalues(tempassign):
            if len(bag) == 2:
                off1, off2 = bag
                first = blk.body[off1]
                second = blk.body[off2]
                inst = ir.Assign(value=first.value, target=second.target,
                                 loc=first.loc)
                # Replacement the second instruction
                blk.body[off2] = inst
                # Remove the first
                removeset.add(off1)

        # Remove from the highest offset to the lowest to preserve order
        for off in reversed(sorted(removeset)):
            del blk.body[off]

    def mark_asssignment(self, tempassign, offset, inst):
        if isinstance(inst, ir.Assign):
            if inst.target.is_temp:
                tempassign[inst.target.name] = [offset]
            elif inst.value.name in tempassign:
                bag = tempassign[inst.value.name]
                if bag[0] == offset - 1:
                    bag.append(offset)
                else:
                    # Only apply to use once temp variable
                    del tempassign[inst.value.name]

