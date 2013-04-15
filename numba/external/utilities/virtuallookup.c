/*

Equivalent of numba.utility.virtuallookup.

Purely here for sanity and debugging reasons.

*/

#include <stdlib.h>
#include "perfecthash.h"

static void *
lookup_method(PyCustomSlots_Table **table_pp,
              uint64_t prehash, char *method_name)
{
    PyCustomSlots_Table *table = *table_pp;
    uint16_t *displacements = (uint16_t *) (
        ((char *) table) + sizeof(PyCustomSlots_Table));
    PyCustomSlots_Entry entry;

    entry = table->entries[((prehash >> table->r) & table->m_f) ^
                           displacements[prehash & table->m_g]];
    if (entry.id == prehash) {
        return entry.ptr;
    } else {
        printf("NumbaError: method '%s' not found\n", method_name);
        abort();
    }
}

static int
export_virtuallookup(PyObject *module)
{
    EXPORT_FUNCTION(lookup_method, module, error)

    return 0;
error:
    return -1;
}
