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

@register
class PerfectHashTableLookup(NumbaCDefinition):
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

    def set_signature(self, env, context):
        table_type = virtual.PyCustomSlots_Table.pointer().pointer()

        self._argtys_ = [
            ('table_slot', table_type.to_llvm(context)),
            ('prehash', uint64.to_llvm(context)),
        ]
        self._retty_ = shortnames.void_p

    def body(self, table_pp, prehash):
        table_p = table_pp.atomic_load('monotonic', align=8)

        #table = table_p.load()
        table_t = from_numba(self.context, virtual.PyCustomSlots_Table)
        table = table_t(self.cbuilder, table_p.handle)

        f = (prehash >> table.r.cast(prehash.type)) & table.m_f
        g = table.d[prehash & table.m_g]
        entry = table.entries[f ^ g]

        with self.ifelse(entry.id == prehash) as ifelse:
            with ifelse.then():
                self.ret(entry.ptr)

            with ifelse.otherwise():
                self.ret(self.constant_null(shortnames.char))

