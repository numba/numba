"""
Virtual method table lookup helpers.

Used to resolve methods from virtual method tables (by lowering
transforms in numba.specialize.exttypes).
"""

from numba import *
from numba import llvm_types
from numba import typedefs
from numba.utility.cbuilder.library import register
from numba.utility.cbuilder.numbacdef import NumbaCDefinition, from_numba
from numba.exttypes import virtual

from llvm_cbuilder import shortnames

#------------------------------------------------------------------------
# Perfect Hash Table Lookup
#------------------------------------------------------------------------

class PerfectHashTableLookup(NumbaCDefinition):
    """
    Look up a method in a PyCustomSlots_Table ** given a prehash.

        PyCustomSlots *vtab = atomic_load(table_pp)

        f = (prehash >> vtab->r) & vtab->m_f
        g = vtab->d[prehash & vtab->m_g]
        PyCustomSlot_Entry *entry = vtab->entries[f ^ g]

        if (entry->id == prehash) {
            void *vmethod = entry.ptr
            call vmethod
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

    def set_signature(self, env, context):
        table_type = virtual.PyCustomSlots_Table.pointer().pointer()

        self._argtys_ = [
            ('table_slot', table_type.to_llvm(context)),
            ('prehash', uint64.to_llvm(context)),
        ]
        self._retty_ = shortnames.pointer(shortnames.void)

    def body(self, table_pp, prehash):
        table_p = table_pp.atomic_load('monotonic', align=8)
        
