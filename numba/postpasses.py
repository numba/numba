# -*- coding: utf-8 -*-

"""
Postpasses over the LLVM IR.
The signature of each postpass is postpass(env, ee, lmod, lfunc) -> lfunc
"""

from __future__ import print_function, division, absolute_import

from numba.support.math_support import math_support, libs

default_postpasses = {}

def register_default(name):
    def dec(f):
        default_postpasses[name] = f
        return f
    return dec

# ______________________________________________________________________
# Postpasses

@register_default('math')
def postpass_link_math(env, ee, lmod, lfunc):
    "numba.math.* -> mathcode.*"
    replacements = {}
    for lf in lmod.functions:
        if lf.name.startswith('numba.math.'):
            _, _, name = lf.name.rpartition('.')
            replacements[lf.name] = name
    del lf # this is dead after linking below

    math_support.link_llvm_math_intrinsics(ee, lmod, libs.math_library,
                                           math_support.LLVMLinker(),
                                           replacements)

    return lfunc
