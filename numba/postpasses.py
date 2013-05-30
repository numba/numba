# -*- coding: utf-8 -*-

"""
Postpasses over the LLVM IR.
The signature of each postpass is postpass(env, ee, lmod, lfunc) -> lfunc
"""

from __future__ import print_function, division, absolute_import

import llvmmath
from llvmmath import linking

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
    "numba.math.* -> llvmmath.*"
    replacements = {}
    for lf in lmod.functions:
        if lf.name.startswith('numba.math.'):
            _, _, name = lf.name.rpartition('.')
            replacements[lf.name] = name
    del lf # this is dead after linking below

    default_math_lib = llvmmath.get_default_math_lib()
    linker = linking.get_linker(default_math_lib)
    linking.link_llvm_math_intrinsics(ee, lmod, default_math_lib,
                                      linker, replacements)
    return lfunc