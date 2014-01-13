# -*- coding: UTF-8 -*-

"""
Capture IR emissions.
"""

from __future__ import print_function, division, absolute_import

import collections
from functools import partial
import llvm.core
from .annotate import SourceIntermediate, Source

# ______________________________________________________________________

class IRBuilder(object):
    def __init__(self, name, builder):
        self.name = name
        self.builder = builder
        self.captured = collections.defaultdict(list)
        self.pos = -1

    def update_pos(self, pos):
        if pos is None:
            pos = -1
        self.pos = pos

    def get_pos(self):
        return self.pos

    def __getattr__(self, attr):
        m = getattr(self.builder, attr)
        if not callable(m):
            return m

        def emit(*args, **kwargs):
            result = m(*args, **kwargs)
            self.captured[self.pos].append(result)
            return result

        return emit

# ______________________________________________________________________

def get_intermediate(ir_builder):
    "Get IR source from an IR builder as a SourceIntermediate"
    linenomap = collections.defaultdict(list)
    linemap = {}
    ir_lineno = 1

    filterer = filters.get(ir_builder.name, lambda x: x)
    ir_builder.captured = filterer(filter_unique(ir_builder.captured))

    for pos, instrs in sorted(ir_builder.captured.iteritems()):
        for instr in instrs:
            linenomap[pos].append(ir_lineno)
            linemap[ir_lineno] = str(instr)
            ir_lineno += 1

    source = Source(linemap, annotations=[])
    return SourceIntermediate(ir_builder.name, linenomap, source)

# ______________________________________________________________________

def filter_llvm(captured):
    for values in captured.values():
        fn = lambda llvm_value: isinstance(llvm_value, llvm.core.Instruction)
        blocks = collections.defaultdict(list)
        for llvm_value in filter(fn, values):
            blocks[llvm_value.basic_block].append(llvm_value)
        values[:] = order_llvm(blocks)
    return captured

def filter_unique(captured):
    for values in captured.values():
        seen = set()
        def unique(item):
            found = item in seen
            seen.add(item)
            return not found

        values[:] = filter(unique, values)
    return captured

# ______________________________________________________________________

def order_llvm(blocks):
    """
    Put llvm instructions and basic blocks in the right order.

    :param blocks: { llvm_block : [llvm_instr] }
    :return: [llvm_line]
    """
    result = []
    if blocks:
        block, values = blocks.popitem()
        blocks[block] = values
        lfunc = block.function

        for block in lfunc.basic_blocks:
            if block in blocks:
                instrs = blocks[block]
                instrpos = dict(
                    (instr, i) for i, instr in enumerate(block.instructions))
                result.append(str(block.name) + ":")
                result.extend(sorted(instrs, key=instrpos.get))

    return result

# ______________________________________________________________________

filters = {
    "llvm": filter_llvm,
}