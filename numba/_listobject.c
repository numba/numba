#include "_listobject.h"

int
numba_list_new(NB_List **out, Py_ssize_t itemsize, Py_ssize_t allocated){
    Py_ssize_t alloc_size = sizeof(NB_List) + itemsize * allocated;
    NB_List *lp = malloc(aligned_size(alloc_size));
    lp->size = 0;
    lp->itemsize = itemsize;
    /* li->items will be allocated as empty */

    *out = lp;
    return 0;
}

Py_ssize_t
numba_list_length(NB_List *lp) {
    return lp->size;
}
