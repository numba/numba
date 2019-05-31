#include "_listobject.h"


static void
copy_item(NB_List *lp, char *dst, const char *src){
    memcpy(dst, src, lp->itemsize);
}

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

int
numba_list_setitem(NB_List *lp, Py_ssize_t index, const char *item) {
    assert(index < lp->size);
    char *loc = lp->items + lp-> itemsize * index;
    copy_item(lp, loc, item);
    return 0;
}
int
numba_list_getitem(NB_List *lp, Py_ssize_t index, char *out) {
    assert(index < lp->size);
    char *loc = lp->items + lp->itemsize * index;
    copy_item(lp, out, loc);
    return 0;
}

int
numba_list_append(NB_List *lp, const char *item) {
    numba_list_setitem(lp, lp->size++, item);
    return 0;
}
