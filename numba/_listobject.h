/* Adapted from CPython3.7 Include/listobject.h */

#ifndef NUMBA_LIST_H
#define NUMBA_LIST_H


typedef struct {
    /* Size of the list.  */
    Py_ssize_t      size;
    /* Size of the list items. */
    Py_ssize_t      itemsize;

    /* items contains space for 'allocated' elements.  The number
     * currently in use is size.
     * Invariants:
     *     0 <= size <= allocated
     *     len(list) == size
     *     item == NULL implies size == allocated == 0
     * FIXME: list.sort() temporarily sets allocated to -1 to detect mutations.
     *
     * Items must normally not be NULL, except during construction when
     * the list is not yet visible outside the function that builds it.
     */
    Py_ssize_t allocated;

    /* Array/pointer for items. Interpretation is governed by itemsize. */
    char  items[];
} NB_List;


NUMBA_EXPORT_FUNC(int)
numba_list_new(NB_List **out, Py_ssize_t itemsize, Py_ssize_t allocated);

NUMBA_EXPORT_FUNC(Py_ssize_t)
numba_list_length(NB_List *lp);

#endif
