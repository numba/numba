# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm

def llvm_alloca(lfunc, builder, ltype, name='', change_bb=True):
    "Use alloca only at the entry bock of the function"
    if change_bb:
        bb = builder.basic_block
    builder.position_at_beginning(lfunc.get_entry_basic_block())
    lstackvar = builder.alloca(ltype, name)
    if change_bb:
        builder.position_at_end(bb)
    return lstackvar

def if_badval(translator, llvm_result, badval, callback,
              cmp=llvm.core.ICMP_EQ, name='cleanup'):
    # Use llvm_cbuilder :(
    b = translator.builder

    bb_true = translator.append_basic_block('%s.if.true' % name)
    bb_endif = translator.append_basic_block('%s.if.end' % name)

    test = b.icmp(cmp, llvm_result, badval)
    b.cbranch(test, bb_true, bb_endif)

    b.position_at_end(bb_true)
    callback(b, bb_true, bb_endif)
    # b.branch(bb_endif)
    b.position_at_end(bb_endif)

    return llvm_result
