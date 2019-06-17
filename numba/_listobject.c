#include "_listobject.h"

typedef enum {
    LIST_OK = 0,
    LIST_ERR_INDEX = -1,
    LIST_ERR_NO_MEMORY = -2,
    LIST_ERR_MUTATED = -3,
    LIST_ERR_ITER_EXHAUSTED = -4,
} ListStatus;

static void
copy_item(NB_List *lp, char *dst, const char *src){
    memcpy(dst, src, lp->itemsize);
}

int
numba_list_new(NB_List **out, Py_ssize_t itemsize, Py_ssize_t allocated){
    NB_List *lp = malloc(aligned_size(sizeof(NB_List)));
    lp->size = 0;
    lp->itemsize = itemsize;
    lp->allocated = allocated;
    memset(&lp->methods, 0x00, sizeof(list_type_based_methods_table));
    lp->items = malloc(aligned_size(lp->itemsize * allocated));

    *out = lp;
    return LIST_OK;
}

static void
list_incref_item(NB_List *lp, const char *item){
    if (lp->methods.item_incref) {
        lp->methods.item_incref(item);
    }
}

static void
list_decref_item(NB_List *lp, const char *item){
    if (lp->methods.item_decref) {
        lp->methods.item_decref(item);
    }
}

void
numba_list_set_method_table(NB_List *lp, list_type_based_methods_table *methods)
{
    memcpy(&lp->methods, methods, sizeof(list_type_based_methods_table));
}

void
numba_list_free(NB_List *lp) {
    /* Clear all references from the item */
    Py_ssize_t i;
    if (lp->methods.item_decref) {
        for (i = 0; i < lp->size; i++) {
            char *item = lp->items + lp->itemsize * i;
            list_decref_item(lp, item);
        }
    }
    free(lp->items);
    free(lp);
}

Py_ssize_t
numba_list_length(NB_List *lp) {
    return lp->size;
}

int
numba_list_setitem(NB_List *lp, Py_ssize_t index, const char *item) {
    if (index >= lp->size) {
        return LIST_ERR_INDEX;
    }
    char *loc = lp->items + lp-> itemsize * index;
    /* This assume there is already an element at index that will be
     * overwritten. DO NOT use this to write to an unassigned location.
     */
    list_decref_item(lp, loc);
    copy_item(lp, loc, item);
    list_incref_item(lp, loc);
    return LIST_OK;
}
int
numba_list_getitem(NB_List *lp, Py_ssize_t index, char *out) {
    if (index >= lp->size) {
        return LIST_ERR_INDEX;
    }
    char *loc = lp->items + lp->itemsize * index;
    copy_item(lp, out, loc);
    return LIST_OK;
}

int
numba_list_append(NB_List *lp, const char *item) {
    int result = numba_list_resize(lp, lp->size + 1);
    if(result < LIST_OK) {
        return result;
    }
    char *loc = lp->items + lp-> itemsize * (lp->size - 1);
    copy_item(lp, loc, item);
    list_incref_item(lp, loc);
    return LIST_OK;
}

int
numba_list_pop(NB_List *lp, Py_ssize_t index, char *out) {
    Py_ssize_t left;
    if (index >= lp->size) {
        return LIST_ERR_INDEX;
    }
    else if (index == lp->size-1) { /* fast path to pop last item */
        char *loc = lp->items + lp->itemsize * index;
        copy_item(lp, out, loc);
        list_decref_item(lp, loc);
        numba_list_resize(lp, lp->size-1);
        return LIST_OK;
    }else{ /* pop from somewhere else */
        /* first get item */
        char *loc = lp->items + lp->itemsize * index;
        copy_item(lp, out, loc);
        list_decref_item(lp, loc);
        /* then incur the dreaded memory copy */
        left = (lp->size - 1 - index) * lp->itemsize;
        char *new_loc = lp->items + lp->itemsize * (index + 1);
        memcpy(loc, new_loc, left);
        numba_list_resize(lp, lp->size-1);
        return LIST_OK;
    }

}

/* Ensure lp->items has room for at least newsize elements, and set
 * lp->size to newsize.  If newsize > lp->size on entry, the content
 * of the new slots at exit is undefined heap trash; it's the caller's
 * responsibility to overwrite them with sane values.
 * The number of allocated elements may grow, shrink, or stay the same.
 * Failure is impossible if newsize <= lp->allocated on entry, although
 * that partly relies on an assumption that the system realloc() never
 * fails when passed a number of bytes <= the number of bytes last
 * allocated (the C standard doesn't guarantee this, but it's hard to
 * imagine a realloc implementation where it wouldn't be true).
 * Note that lp->items may change, and even if newsize is less
 * than lp->size on entry.
 */
int
numba_list_resize(NB_List *lp, Py_ssize_t newsize) {
    char * items;
    size_t new_allocated, num_allocated_bytes;
    /* Bypass realloc() when a previous overallocation is large enough
       to accommodate the newsize.  If the newsize falls lower than half
       the allocated size, then proceed with the realloc() to shrink the list.
    */
    if (lp->allocated >= newsize && newsize >= (lp->allocated >> 1)) {
        assert(lp->items != NULL || newsize == 0);
        lp->size = newsize;
        return 0;
    }
    /* This over-allocates proportional to the list size, making room
     * for additional growth.  The over-allocation is mild, but is
     * enough to give linear-time amortized behavior over a long
     * sequence of appends() in the presence of a poorly-performing
     * system realloc().
     * The growth pattern is:  0, 4, 8, 16, 25, 35, 46, 58, 72, 88, ...
     * Note: new_allocated won't overflow because the largest possible value
     *       is PY_SSIZE_T_MAX * (9 / 8) + 6 which always fits in a size_t.
     */
    new_allocated = (size_t)newsize + (newsize >> 3) + (newsize < 9 ? 3 : 6);
    if (new_allocated > (size_t)PY_SSIZE_T_MAX / lp->itemsize) {
        return LIST_ERR_NO_MEMORY;
    }

    if (newsize == 0)
        new_allocated = 0;
    num_allocated_bytes = new_allocated * lp->itemsize;
    items = realloc(lp->items, aligned_size(num_allocated_bytes));
    if (!items) {
        return LIST_ERR_NO_MEMORY;
    }
    lp->items = items;
    lp->size = newsize;
    lp->allocated = (Py_ssize_t)new_allocated;
    return LIST_OK;
}


size_t
numba_list_iter_sizeof() {
    return sizeof(NB_ListIter);
}

void
numba_list_iter(NB_ListIter *it, NB_List *l) {
    it->parent = l;
    it->size = l->size;
    it->pos = 0;
}

int
numba_list_iter_next(NB_ListIter *it, const char **item_ptr) {
    NB_List *lp = it->parent;
    /* FIXME: Detect list mutation during iteration */
    if (lp->size != it->size) {
        return LIST_ERR_MUTATED;
    }
    if (it->pos < lp->size) {
        *item_ptr = lp->items + lp->itemsize * it->pos++;
        return OK;
    }else{
        return LIST_ERR_ITER_EXHAUSTED;
    }
}


#define CHECK(CASE) {                                                   \
    if ( !(CASE) ) {                                                    \
        printf("'%s' failed file %s:%d\n", #CASE, __FILE__, __LINE__);   \
        return -1;                                                       \
    }                                                                   \
}

int
numba_test_list(void) {
    NB_List *lp;
    int status;
    Py_ssize_t it_count;
    const char *it_item;
    NB_ListIter iter;
    char got_item[4];

    puts("test_list");


    status = numba_list_new(&lp, 4, 0);
    CHECK(status == LIST_OK);
    CHECK(lp->itemsize == 4);
    CHECK(lp->size == 0);
    CHECK(lp->allocated == 0);

    // insert 1st item, this will cause a realloc
    status = numba_list_append(lp, "abc");
    CHECK(status == LIST_OK);
    CHECK(lp->size == 1);
    CHECK(lp->allocated == 4);
    status = numba_list_getitem(lp, 0, got_item);
    CHECK(status == LIST_OK);
    CHECK(memcmp(got_item, "abc", 4) == 0);

    // insert 2nd item
    status = numba_list_append(lp, "def");
    CHECK(status == LIST_OK);
    CHECK(lp->size == 2);
    CHECK(lp->allocated == 4);
    status = numba_list_getitem(lp, 1, got_item);
    CHECK(status == LIST_OK);
    CHECK(memcmp(got_item, "def", 4) == 0);

    // insert 3rd item
    status = numba_list_append(lp, "ghi");
    CHECK(status == LIST_OK);
    CHECK(lp->size == 3);
    CHECK(lp->allocated == 4);
    status = numba_list_getitem(lp, 2, got_item);
    CHECK(status == LIST_OK);
    CHECK(memcmp(got_item, "ghi", 4) == 0);

    // insert 4th item
    status = numba_list_append(lp, "jkl");
    CHECK(status == LIST_OK);
    CHECK(lp->size == 4);
    CHECK(lp->allocated == 4);
    status = numba_list_getitem(lp, 3, got_item);
    CHECK(status == LIST_OK);
    CHECK(memcmp(got_item, "jkl", 4) == 0);

    // insert 5th item, this will cause another realloc
    status = numba_list_append(lp, "mno");
    CHECK(status == LIST_OK);
    CHECK(lp->size == 5);
    CHECK(lp->allocated == 8);
    status = numba_list_getitem(lp, 4, got_item);
    CHECK(status == LIST_OK);
    CHECK(memcmp(got_item, "mno", 4) == 0);

    // Overwrite 1st item
    status = numba_list_setitem(lp, 0, "pqr");
    CHECK(status == LIST_OK);
    CHECK(lp->size == 5);
    CHECK(lp->allocated == 8);
    status = numba_list_getitem(lp, 0, got_item);
    CHECK(status == LIST_OK);
    CHECK(memcmp(got_item, "pqr", 4) == 0);

    // Pop 1st item, check item shift
    status = numba_list_pop(lp, 0, got_item);
    CHECK(status == LIST_OK);
    CHECK(lp->size == 4);
    CHECK(lp->allocated == 8);
    CHECK(memcmp(got_item, "pqr", 4) == 0);
    CHECK(memcmp(lp->items, "def\x00ghi\x00jkl\x00mno\x00", 16) == 0);

    // Pop last (4th) item, no shift since only last item affected
    status = numba_list_pop(lp, 3, got_item);
    CHECK(status == LIST_OK);
    CHECK(lp->size == 3);
    CHECK(lp->allocated == 6);  // this also shrinks the allocation
    CHECK(memcmp(got_item, "mno", 4) == 0);
    CHECK(memcmp(lp->items, "def\x00ghi\x00jkl\x00", 12) == 0);


    // Test iterator
    CHECK(lp->size > 0);
    numba_list_iter(&iter, lp);
    it_count = 0;
    CHECK(iter.parent == lp);
    CHECK(iter.pos == it_count);

    // Current contents of list
    const char items[] = "def\x00ghi\x00jkl\x00";
    while ( (status = numba_list_iter_next(&iter, &it_item)) == OK) {
        it_count += 1;
        CHECK(iter.pos == it_count); // check iterator position
        CHECK(it_item != NULL); // quick check item is non-null
        // go fishing in items
        CHECK(memcmp((const char *)items + ((it_count-1) * 4), it_item, 4) == 0);
    }

    CHECK(status == LIST_ERR_ITER_EXHAUSTED);
    CHECK(lp->size == it_count);

    // free existing list
    numba_list_free(lp);

    // test growth upon append and shrink during pop
    status = numba_list_new(&lp, 1, 0);
    CHECK(status == LIST_OK);
    CHECK(lp->itemsize == 1);
    CHECK(lp->size == 0);
    CHECK(lp->allocated == 0);

    // first, grow the list
    uint8_t i;
    // Use exactly 16 elements, should go through the allocation pattern:
    // 0, 4, 8, 16, 25
    for (i = 0; i < 17 ; i++) {
        switch(i) {
            // Check the allocation before
            case 0:  CHECK(lp->allocated == 0); break;
            case 4:  CHECK(lp->allocated == 4); break;
            case 8:  CHECK(lp->allocated == 8); break;
            case 16: CHECK(lp->allocated == 16); break;
        }
        // To insert a single byte element into the list, dereference it to get
        // the point to it's value and then cast that pointer to a
        // (const char *) so that append will accept it.
        status = numba_list_append(lp, (const char*)&i);
        CHECK(status == LIST_OK);
        switch(i) {
            // Check that the growth happend accordingly
            case 0:  CHECK(lp->allocated == 4); break;
            case 4:  CHECK(lp->allocated == 8); break;
            case 8:  CHECK(lp->allocated == 16); break;
            case 16: CHECK(lp->allocated == 25); break;
        }
    }
    CHECK(lp->size == 17);

    // Check current contents of list
    const char growth_items[] = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10";
    CHECK(memcmp(lp->items, growth_items, 17) == 0);

    // Now, pop them again and check that list shrinks
    char received[1];
    for (i = 17; i > 0 ; i--) {
        switch(i) {
             // Check the allocation before pop
             case 17:  CHECK(lp->allocated == 25); break;
             case 12:  CHECK(lp->allocated == 25); break;
             case 9:   CHECK(lp->allocated == 18); break;
             case 6:   CHECK(lp->allocated == 12); break;
             case 4:   CHECK(lp->allocated == 8); break;
             case 3:   CHECK(lp->allocated == 6); break;
             case 2:   CHECK(lp->allocated == 5); break;
             case 1:   CHECK(lp->allocated == 4); break;
        }
        status = numba_list_pop(lp, i-1, received);
        CHECK(status == LIST_OK);
        switch(i) {
             // Check that the shrink happend accordingly
             case 17:  CHECK(lp->allocated == 25); break;
             case 12:  CHECK(lp->allocated == 18); break;
             case 9:   CHECK(lp->allocated == 12); break;
             case 6:   CHECK(lp->allocated == 8); break;
             case 4:   CHECK(lp->allocated == 6); break;
             case 3:   CHECK(lp->allocated == 5); break;
             case 2:   CHECK(lp->allocated == 4); break;
             case 1:   CHECK(lp->allocated == 0); break;
        }
    }

    // free list and return 0
    numba_list_free(lp);
    return 0;

}

#undef CHECK
