# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from numba.typesystem import typesystem, universe

import llvm.core
pointer = llvm.core.Type.pointer

lluniverse = universe.LowLevelUniverse()
llvm_universe = universe.LLVMUniverse()
numba_universe = universe.NumbaUniverse()

ts = typesystem.TypeSystem(numba_universe)

def test_llvm_conversion():
    print(numba_universe.int, llvm_universe.int)
