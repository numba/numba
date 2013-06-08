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
        self.pos = None

    def update_pos(self, pos):
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
    "Get annotations from an IR builder"
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
        newvals = []
        block = None
        for llvm_value in filter(fn, values):
            if llvm_value.basic_block != block:
                newvals.append(llvm_value.basic_block.name + ":")
                block = llvm_value.basic_block
            newvals.append(llvm_value)
            # if llvm_value.opcode_name == 'br':
            #     print(llvm_value.basic_block)
        values[:] = newvals
    return captured

def findoradd(seen, val):
    found = val in seen
    seen.add(val)
    return not found

def filter_unique(captured):
    for values in captured.values():
        values[:] = filter(partial(findoradd, set()), values)
    return captured

# ______________________________________________________________________

filters = {
    "llvm": filter_llvm,
}