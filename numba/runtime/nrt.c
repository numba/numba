#include <stdarg.h>
#include <string.h> /* for memset */
#include "nrt.h"
#include "assert.h"

#if !defined MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif

#ifdef IS_FLAG_SET
#  error "IS_FLAG_SET redefined"
#else
#  define IS_FLAG_SET(X, M) ( (X & M) == M )
#endif

typedef int (*atomic_meminfo_cas_func)(void **ptr, void *cmp,
                                       void *repl, void **oldptr);


enum MEMINFO_FLAGS{
    MEMINFO_FLAGS_NONE = 0,
    /*
     *  If set, the MemInfo and the data buffer are in the same allocated
     *  region.  Freeing the MemInfo pointer will free the data.
     */
    MEMINFO_FLAGS_INLINED = 1
};

union MemInfo{
    struct {
        size_t         refct;
        dtor_function  dtor;
        void          *dtor_info;
        void          *data;
        size_t         size;    /* only used for NRT allocated memory */
        int            flags;   /* See enum MEMINFO_FLAGS */
    } payload;

    /* Freelist or Deferred-dtor-list */
    MemInfo *list_next;
};


typedef struct MIList {
    MemInfo * volatile head;
    void              *lock;
} MIList;

#define MILIST_UNLOCKED ((void*)0)
#define MILIST_LOCKED ((void*)1)


struct MemSys{
    /* Ununsed MemInfo are recycled here */
    MIList mi_freelist;
    /* MemInfo with deferred dtor */
    MIList  mi_deferlist;
    /* Atomic increment and decrement function */
    atomic_inc_dec_func atomic_inc, atomic_dec;
    /* Atomic CAS */
    atomic_meminfo_cas_func atomic_cas;
    /* Shutdown flag */
    int shutting;
    /* Stats */
    size_t stats_alloc, stats_free, stats_mi_alloc, stats_mi_free;

};

/* The Memory System object */
static MemSys TheMSys;

static
void milist_lock(MIList *list) {
    void *dummy;
    while(!TheMSys.atomic_cas(&list->lock, MILIST_UNLOCKED, MILIST_LOCKED,
                              &dummy));
}

static
void milist_unlock(MIList *list) {
    void *dummy;
    while(!TheMSys.atomic_cas(&list->lock, MILIST_LOCKED, MILIST_UNLOCKED,
                              &dummy));
}

static
MemInfo *nrt_pop_meminfo_list(MIList *list) {
    MemInfo *repl, *head;

    milist_lock(list);

    head = list->head;     /* get the current head */
    if ( head ) {
        /* if head is not NULL, replace with the next item */
        repl = head->list_next;
    } else {
        /* else, replace with NULL */
        repl = NULL;
    }
    list->head = repl;

    milist_unlock(list);
    return head;
}

static
void nrt_push_meminfo_list(MIList *list, MemInfo *repl) {
    MemInfo *old, *head;

    milist_lock(list);

    head = list->head;   /* get the current head */
    /* Set the next item to be the current head */
    repl->list_next = head;
    /* Set new head */
    list->head = repl;

    milist_unlock(list);
}

static
void nrt_meminfo_call_dtor(MemInfo *mi) {
    NRT_Debug(nrt_debug_print("nrt_meminfo_call_dtor %p\n", mi));
    /* call dtor */
    if (mi->payload.dtor)
        mi->payload.dtor(mi->payload.data, mi->payload.dtor_info);
    /* Clear and release MemInfo */
    NRT_MemInfo_destroy(mi);
}

static
MemInfo* meminfo_malloc(void) {
    void *p = malloc(sizeof(MemInfo));;
    NRT_Debug(nrt_debug_print("meminfo_malloc %p\n", p));
    return p;
}

void NRT_MemSys_init(void) {
    memset(&TheMSys, 0, sizeof(MemSys));
}

void NRT_MemSys_shutdown(void) {
    TheMSys.shutting = 1;
    /* Revert to use our non-atomic stub for all atomic operations
       because the JIT-ed version will be removed.
       Since we are at interpreter shutdown,
       it cannot be running multiple threads anymore. */
    NRT_MemSys_set_atomic_inc_dec_stub();
    NRT_MemSys_set_atomic_cas_stub();
}

void NRT_MemSys_process_defer_dtor(void) {
    MemInfo *mi;
    while ((mi = nrt_pop_meminfo_list(&TheMSys.mi_deferlist))) {
        NRT_Debug(nrt_debug_print("Defer dtor %p\n", mi));
        nrt_meminfo_call_dtor(mi);
    }
}

void NRT_MemSys_insert_meminfo(MemInfo *newnode) {
    assert(newnode && "`newnode` cannot be NULL");
    /*
    if (NULL == newnode) {
        newnode = meminfo_malloc();
    } else {
        assert(newnode->payload.refct == 0 && "RefCt must be 0");
    }
    */
    assert(newnode->payload.refct == 0 && "RefCt must be 0");
    NRT_Debug(nrt_debug_print("NRT_MemSys_insert_meminfo newnode=%p\n",
                              newnode));
    memset(newnode, 0, sizeof(MemInfo));  /* to catch bugs; not required */
    nrt_push_meminfo_list(&TheMSys.mi_freelist, newnode);
}

MemInfo* NRT_MemSys_pop_meminfo(void) {
    MemInfo *node = nrt_pop_meminfo_list(&TheMSys.mi_freelist);
    if (NULL == node) {
        node = meminfo_malloc();
    }
    memset(node, 0, sizeof(MemInfo));   /* to catch bugs; not required */
    NRT_Debug(nrt_debug_print("NRT_MemSys_pop_meminfo: return %p\n", node));
    return node;
}

void NRT_MemSys_set_atomic_inc_dec(atomic_inc_dec_func inc,
                                   atomic_inc_dec_func dec)
{
    TheMSys.atomic_inc = inc;
    TheMSys.atomic_dec = dec;
}

void NRT_MemSys_set_atomic_cas(atomic_cas_func cas) {
    TheMSys.atomic_cas = (atomic_meminfo_cas_func)cas;
}

size_t NRT_MemSys_get_stats_alloc() {
    return TheMSys.stats_alloc;
}

size_t NRT_MemSys_get_stats_free() {
    return TheMSys.stats_free;
}

size_t NRT_MemSys_get_stats_mi_alloc() {
    return TheMSys.stats_mi_alloc;
}

size_t NRT_MemSys_get_stats_mi_free() {
    return TheMSys.stats_mi_free;
}

static
size_t nrt_testing_atomic_inc(size_t *ptr){
    /* non atomic */
    size_t out = *ptr;
    out += 1;
    *ptr = out;
    return out;
}

static
size_t nrt_testing_atomic_dec(size_t *ptr){
    /* non atomic */
    size_t out = *ptr;
    out -= 1;
    *ptr = out;
    return out;
}

static
int nrt_testing_atomic_cas(void* volatile *ptr, void *cmp, void *val,
                           void * *oldptr){
    /* non atomic */
    void *old = *ptr;
    *oldptr = old;
    if (old == cmp) {
        *ptr = val;
         return 1;
    }
    return 0;

}

void NRT_MemSys_set_atomic_inc_dec_stub(void){
    NRT_MemSys_set_atomic_inc_dec(nrt_testing_atomic_inc,
                                  nrt_testing_atomic_dec);
}

void NRT_MemSys_set_atomic_cas_stub(void) {
    NRT_MemSys_set_atomic_cas(nrt_testing_atomic_cas);
}

void NRT_MemInfo_init(MemInfo *mi,void *data, size_t size, dtor_function dtor,
                      void *dtor_info, int flags)
{
    mi->payload.refct = 1;  /* starts with 1 refct */
    mi->payload.dtor = dtor;
    mi->payload.dtor_info = dtor_info;
    mi->payload.data = data;
    mi->payload.size = size;
    mi->payload.flags = flags;
    /* Update stats */
    TheMSys.atomic_inc(&TheMSys.stats_mi_alloc);
}

MemInfo* NRT_MemInfo_new(void *data, size_t size, dtor_function dtor,
                         void *dtor_info)
{
    MemInfo * mi = NRT_MemSys_pop_meminfo();
    NRT_MemInfo_init(mi, data, size, dtor, dtor_info, MEMINFO_FLAGS_NONE);
    return mi;
}

size_t NRT_MemInfo_refcount(MemInfo *mi) {
    /* Should never returns 0 for a valid MemInfo */
    if (mi && mi->payload.data)
        return mi->payload.refct;
    else{
        return (size_t)-1;
    }
}

static
void nrt_internal_dtor_safe(void *ptr, void *info) {
    size_t size = (size_t) info;
    NRT_Debug(nrt_debug_print("nrt_internal_dtor_safe %p, %p\n", ptr, info));
    /* See NRT_MemInfo_alloc_safe() */
    memset(ptr, 0xDE, MIN(size, 256));
}

static
void *nrt_allocate_meminfo_and_data(size_t size, MemInfo **mi_out) {
    MemInfo *mi;
    char *base = NRT_Allocate(sizeof(MemInfo) + size);
    mi = (MemInfo*)base;
    *mi_out = mi;
    return base + sizeof(MemInfo);
}

MemInfo* NRT_MemInfo_alloc(size_t size) {
    MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data(size, &mi);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc %p\n", data));
    NRT_MemInfo_init(mi, data, size, NULL, NULL, MEMINFO_FLAGS_INLINED);
    return mi;
}

MemInfo* NRT_MemInfo_alloc_safe(size_t size) {
    MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data(size, &mi);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_safe %p %zu\n", data, size));
    NRT_MemInfo_init(mi, data, size, nrt_internal_dtor_safe,
                     (void*)size, MEMINFO_FLAGS_INLINED);
    return mi;
}

static
void* nrt_allocate_meminfo_and_data_align(size_t size, unsigned align,
                                         MemInfo **mi)
{
    size_t offset, intptr, remainder;
    char *base = nrt_allocate_meminfo_and_data(size + align, mi);
    intptr = (size_t) base;
    /* See if we are aligned */
    remainder = intptr % align;
    if (remainder == 0){ /* Yes */
        offset = 0;
    } else { /* No, move forward `offset` bytes */
        offset = align - remainder;
    }
    return base + offset;
}

MemInfo* NRT_MemInfo_alloc_aligned(size_t size, unsigned align) {
    MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data_align(size, align, &mi);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_aligned %p\n", data));
    NRT_MemInfo_init(mi, data, size, NULL, NULL, MEMINFO_FLAGS_INLINED);
    return mi;
}

MemInfo* NRT_MemInfo_alloc_safe_aligned(size_t size, unsigned align) {
    MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data_align(size, align, &mi);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_safe_aligned %p %zu\n",
                              data, size));
    NRT_MemInfo_init(mi, data, size, nrt_internal_dtor_safe,
                     (void*)size, MEMINFO_FLAGS_INLINED);
    return mi;
}

void NRT_MemInfo_destroy(MemInfo *mi) {
    if (IS_FLAG_SET(mi->payload.flags, MEMINFO_FLAGS_INLINED)) {
        NRT_Free(mi);
    }
    else {
        NRT_MemSys_insert_meminfo(mi);
    }
    TheMSys.atomic_inc(&TheMSys.stats_mi_free);
}

void NRT_MemInfo_acquire(MemInfo *mi) {
    NRT_Debug(nrt_debug_print("NRT_acquire %p refct=%zu\n", mi,
                                                            mi->payload.refct));
    assert(mi->payload.refct > 0 && "RefCt cannot be zero");
    TheMSys.atomic_inc(&mi->payload.refct);
}

void NRT_MemInfo_call_dtor(MemInfo *mi, int defer) {
    /* We have a destructor */
    if (defer) {
        NRT_MemInfo_defer_dtor(mi);
    } else {
        nrt_meminfo_call_dtor(mi);
    }
}

void NRT_MemInfo_release(MemInfo *mi, int defer) {
    NRT_Debug(nrt_debug_print("NRT_release %p refct=%zu\n", mi,
                                                            mi->payload.refct));
    assert (mi->payload.refct > 0 && "RefCt cannot be 0");
    /* RefCt drop to zero */
    if (TheMSys.atomic_dec(&mi->payload.refct) == 0) {
        NRT_MemInfo_call_dtor(mi, defer);
    }
}

void* NRT_MemInfo_data(MemInfo* mi) {
    return mi->payload.data;
}

size_t NRT_MemInfo_size(MemInfo* mi) {
    return mi->payload.size;
}

void NRT_MemInfo_defer_dtor(MemInfo *mi) {
    NRT_Debug(nrt_debug_print("NRT_MemInfo_defer_dtor\n"));
    nrt_push_meminfo_list(&TheMSys.mi_deferlist, mi);
}

void NRT_MemInfo_dump(MemInfo *mi, FILE *out) {
    fprintf(out, "MemInfo %p refcount %zu\n", mi, mi->payload.refct);
}

void* NRT_Allocate(size_t size) {
    void *ptr = malloc(size);
    NRT_Debug(nrt_debug_print("NRT_Allocate bytes=%llu ptr=%p\n", size, ptr));
    TheMSys.atomic_inc(&TheMSys.stats_alloc);
    return ptr;
}

void NRT_Free(void *ptr) {
    NRT_Debug(nrt_debug_print("NRT_Free %p\n", ptr));
    free(ptr);
    TheMSys.atomic_inc(&TheMSys.stats_free);
}


