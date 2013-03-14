"""
Virtual method lookup written in Numba.
"""

import ctypes

import numba
from numba import *
from numba.exttypes import virtual


table_t = virtual.PyCustomSlots_Table

table_t_pp = table_t.pointer().pointer()
char_p = char.pointer()
void_p = void.pointer()
uint16_p = uint16.pointer()

# This is a bad idea!
# displacements_offset = table_t.offsetof('d')
displacements_offset = ctypes.sizeof(table_t.to_ctypes())

@jit(void_p(table_t_pp, uint64), wrap=False)
def lookup_method(table_pp, prehash):
    """
    Look up a method in a PyCustomSlots_Table ** given a prehash.

        PyCustomSlots *vtab = atomic_load(table_pp)

        f = (prehash >> vtab->r) & vtab->m_f
        g = vtab->d[prehash & vtab->m_g]
        PyCustomSlot_Entry *entry = vtab->entries[f ^ g]

        if (entry->id == prehash) {
            void *vmethod = entry.ptr
            call vmethod(obj, ...)
        } else {
            PyObject_Call(obj, "method", ...)
        }

    Note how the object stores a vtable **, instead of a vtable *. This
    indirection allows producers of the table to generate a new table
    (e.g. after adding a specialization) and replace the table for all
    live objects.

    We then atomically load the table, to allow using the table without
    the GIL (and having the GIL holding thread update the table and rewrite
    the pointer).

    Hence the table pointer should *not* be cached by callers, since the
    first table miss can trigger a compilation you will want to find the
    second time around.
    """
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
        # print "found!"
        return entry.ptr
    else:
        print "not found :("
        return numba.NULL
