"""
Virtual method lookup written in Numba.
"""

import numba
from numba import *
from numba.exttypes import virtual


table_t = virtual.PyCustomSlots_Table

table_t_pp = table_t.pointer().pointer()
char_p = char.pointer()
void_p = void.pointer()
uint16_p = uint16.pointer()

displacements_offset = table_t.offsetof('d')


@jit(void_p(table_t_pp, uint64), wrap=False)
def lookup_method(table_pp, prehash):
    table_p = table_pp[0]
    table = table_p[0]
    displacements = uint16_p(char_p(table_p) + displacements_offset)

    # Compute f
    f = (prehash >> table.r) & table.m_f

    # Compute g
    # g = table.d[prehash & table.m_g]
    g = displacements[prehash & table.m_g]

    entry = table.entries[f ^ g]

    if entry.id == prehash:
        return entry.ptr
    else:
        return numba.NULL