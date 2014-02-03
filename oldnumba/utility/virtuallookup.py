# -*- coding: utf-8 -*-

"""
Virtual method lookup written in Numba.
"""

from __future__ import division, absolute_import

import ctypes.util

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

#------------------------------------------------------------------------
# Some libc functions
#------------------------------------------------------------------------

libc = ctypes.CDLL(ctypes.util.find_library('c'))

abort = libc.abort
abort.restype = None
abort.argtypes = []

printf = libc.printf
printf.restype = ctypes.c_int
printf.argtypes = [ctypes.c_char_p]

@jit(void_p(table_t_pp, uint64), wrap=False, nopython=True)
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
        return entry.ptr
    else:
        return numba.NULL

@jit(void_p(table_t_pp, uint64, char_p), wrap=False, nopython=True)
def lookup_and_assert_method(table_pp, prehash, method_name):
    result = lookup_method(table_pp, prehash)
    if result == numba.NULL:
        # printf("Error: expected method %s to be available\n", method_name)
        # print "Error: expected method", method_name,  "to be available"
        printf("NumbaError: expected method ")
        printf(method_name)
        printf(" to be available\n")
        abort()

    return result
