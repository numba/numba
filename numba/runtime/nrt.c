#include <stdarg.h>
#include <string.h> /* for memset */
#include "nrt.h"
#include "assert.h"

#if !defined MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif

typedef size_t word_type;

typedef int (*atomic_meminfo_cas_func)(volatile word_type *ptr, word_type cmp,
                                       word_type repl,
                                       volatile word_type *oldptr);


union MemInfo{
    struct {
        size_t         refct;
        dtor_function  dtor;
        void          *dtor_info;
        void          *data;
        size_t         size;    /* only used for NRT allocated memory */
    } payload;

    /* Freelist or Deferred-dtor-list */
    MemInfo *list_next;
};


typedef struct MemInfoList {
    MemInfo * volatile list;
    volatile word_type state;
} MemInfoList;

#define UNLOCK ((word_type)0)
#define LOCK_PUSH ((word_type)1 << 31)
#define LOCK_POP ((word_type)1 << 30)
#define LOCK_COUNT_MASK ((word_type)0xFFFF)


struct MemSys{
    /* Ununsed MemInfo are recycled here */
    MemInfoList mi_freelist;
    /* MemInfo with deferred dtor */
    MemInfoList mi_deferlist;
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


typedef struct {
    size_t total_size;
} AlignHeader;


static
void list_lock(MemInfoList *listst, word_type mask) {
    volatile word_type actual;
    word_type old;
    word_type newval;

    /* First load */
    actual = listst->state;

    while (1) {
        /* `actual` is loaded in the loop by atomic_cas */
        old = actual;

        /* If in unlocked state, set to init "mask" state */
        if ( old == UNLOCK ) newval = mask;
        /* If in mask state, increment */
        else if ( (old & mask) == mask
                  && (old & LOCK_COUNT_MASK) < LOCK_COUNT_MASK )
            newval = old + 1;
        /* If not in mask state, load and retry */
        else {
            actual = listst->state;
            continue;
        }

        /* CAS with atomic load to actual */
        if (TheMSys.atomic_cas(&listst->state, old, newval, &actual)) {
            /* Succeed, quit */
            return;
        }
    }
}

static
void list_unlock(MemInfoList *listst, word_type mask) {
    volatile word_type actual;
    word_type old;
    word_type newval;

    /* First load */
    actual = listst->state;

    while (1) {
        /* `actual` is loaded in the loop by atomic_cas */
        old = actual;

        /* If in init "mask" state, set to unlock state */
        if ( old == mask ) newval = UNLOCK;
        /* If in mask state, decrement */
        else if ( (old & mask) == mask
                   && (old & LOCK_COUNT_MASK) <= LOCK_COUNT_MASK)
           newval = old - 1;
        /* If non "mask" state, fatal error */
        else {
            puts("Corrupted list state");
            exit(1);
        }

        /* CAS with atomic load to `actual` */
        if (TheMSys.atomic_cas(&listst->state, old, newval, &actual)) {
            /* Succeed, quit */
            return;
        }
    }
}

static
MemInfo *nrt_pop_meminfo_list(MemInfoList *listst) {
    MemInfo *old, *repl, *head;
    MemInfo * volatile * volatile list = &listst->list;

    list_lock(listst, LOCK_POP);

    head = *list;     /* get the current head */
    do {
        old = head;   /* old is what CAS compare against */
        if ( head ) {
            /* if head is not NULL, replace with the next item */
            repl = head->list_next;
        } else {
            /* else, replace with NULL */
            repl = NULL;
        }
        /* Try to replace list head with the next node.
           The function also perform:
               head <- atomicload(list) */
    } while ( !TheMSys.atomic_cas((word_type*)list,
                                  (word_type)old,
                                  (word_type)repl,
                                  (word_type*)&head));


    list_unlock(listst, LOCK_POP);
    return old;
}

static
void nrt_push_meminfo_list(MemInfoList *listst, MemInfo *repl) {
    MemInfo *old, *head;
    MemInfo * volatile * volatile list = &listst->list;

    list_lock(listst, LOCK_PUSH);

    head = *list;   /* get the current head */
    do {
        old = head; /* old is what CAS compare against */
        /* Set the next item to be the current head */
        repl->list_next = head;
        /* Try to replace the head with the new node.
           The function also perform:
               head <- atomicload(list) */
    } while ( !TheMSys.atomic_cas((word_type*)list,
                                  (word_type)old,
                                  (word_type)repl,
                                  (word_type*)&head));

    list_unlock(listst, LOCK_PUSH);
}

static
void nrt_meminfo_call_dtor(MemInfo *mi) {
    NRT_Debug(nrt_debug_print("nrt_meminfo_call_dtor %p\n", mi));
    /* call dtor */
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
    TheMSys.atomic_inc(&TheMSys.stats_mi_free);
}

MemInfo* NRT_MemSys_pop_meminfo(void) {
    MemInfo *node = nrt_pop_meminfo_list(&TheMSys.mi_freelist);
    if (NULL == node) {
        node = meminfo_malloc();
    }
    memset(node, 0, sizeof(MemInfo));   /* to catch bugs; not required */
    NRT_Debug(nrt_debug_print("NRT_MemSys_pop_meminfo: return %p\n", node));
    TheMSys.atomic_inc(&TheMSys.stats_mi_alloc);
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

MemInfo* NRT_MemInfo_new(void *data, size_t size, dtor_function dtor,
                         void *dtor_info)
{
    MemInfo * mi = NRT_MemSys_pop_meminfo();
    mi->payload.refct = 1;  /* starts with 1 refct */
    mi->payload.dtor = dtor;
    mi->payload.dtor_info = dtor_info;
    mi->payload.data = data;
    mi->payload.size = size;
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
void nrt_internal_dtor(void *ptr, void *info) {
    NRT_Debug(nrt_debug_print("nrt_internal_dtor %p, %p\n", ptr, info));
    NRT_Free(ptr);
}

static
void nrt_internal_dtor_safe(void *ptr, void *info) {
    size_t size = (size_t) info;
    NRT_Debug(nrt_debug_print("nrt_internal_dtor_safe %p, %p\n", ptr, info));
    /* See NRT_MemInfo_alloc_safe() */
    memset(ptr, 0xDE, MIN(size, 256));
    NRT_Free(ptr);
}

MemInfo* NRT_MemInfo_alloc(size_t size) {
    void *data = NRT_Allocate(size);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc %p\n", data));
    return NRT_MemInfo_new(data, size, nrt_internal_dtor, NULL);
}

MemInfo* NRT_MemInfo_alloc_safe(size_t size) {
    void *data = NRT_Allocate(size);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_safe %p %llu\n", data, size));
    return NRT_MemInfo_new(data, size, nrt_internal_dtor_safe, (void*)size);
}

static
void nrt_internal_aligned_dtor(void *ptr, void *info) {
    NRT_Debug(nrt_debug_print("nrt_internal_aligned_dtor %p, %p\n", ptr, info));
    NRT_Free(info);
}

static
void nrt_internal_aligned_safe_dtor(void *ptr, void *info) {
    AlignHeader *header = info;
    NRT_Debug(nrt_debug_print("nrt_internal_aligned_safe_dtor %p, %p\n", ptr,
                              info));
    if (header->total_size) {
        memset(header, 0xDE, header->total_size);  /* for safety */
    }
    NRT_Free(info);
}

MemInfo* NRT_MemInfo_alloc_aligned(size_t size, unsigned align) {
    void *data = NULL;
    void *base = NRT_MemAlign(&data, size, align);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_aligned %p\n", data));
    return NRT_MemInfo_new(data, size, nrt_internal_aligned_dtor, base);
}

MemInfo* NRT_MemInfo_alloc_safe_aligned(size_t size, unsigned align) {
    void *data = NULL;
    void *base = NRT_MemAlign(&data, size, align);
    memset(data, 0xCB, size);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_safe_aligned %p %llu\n",
                              data, size));
    return NRT_MemInfo_new(data, size, nrt_internal_aligned_safe_dtor, base);
}

void NRT_MemInfo_destroy(MemInfo *mi) {
    NRT_MemSys_insert_meminfo(mi);
}

void NRT_MemInfo_acquire(MemInfo *mi) {
    NRT_Debug(nrt_debug_print("NRT_acquire %p refct=%zu\n", mi,
                                                            mi->payload.refct));
    assert(mi->payload.refct > 0 && "RefCt cannot be zero");
    TheMSys.atomic_inc(&mi->payload.refct);
}

void NRT_MemInfo_call_dtor(MemInfo *mi, int defer) {
    /* We have a destructor */
    if (mi->payload.dtor) {
        if (defer) {
            NRT_MemInfo_defer_dtor(mi);
        } else {
            nrt_meminfo_call_dtor(mi);
        }
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

void* NRT_MemAlign(void **ptr, size_t size, unsigned align) {
    AlignHeader *base;
    size_t intptr;
    unsigned offset;
    unsigned remainder;

    /* Allocate extra space for padding and book keeping */
    size_t total_size = size + 2 * align + sizeof(AlignHeader);
    base = (AlignHeader*) NRT_Allocate(total_size);

    /* The AlignHeader goes first, so skip sizeof(AlignHeader) */
    intptr = (size_t) (base + 1);

    /* See if we are aligned */
    remainder = intptr % align;
    if (remainder == 0){
        /* Yes */
        offset = 0;
    } else {
        /* No, move forward `offset` bytes */
        offset = align - remainder;
    }

    /* Store the aligned pointer to the output parameter */
    *ptr = (void*) (intptr + offset);

    /* Remember the total allocated size */
    base->total_size = total_size;

    /* Return the pointer for deallocation with NRT_Free() */
    return base;
}
