#include <stdarg.h>
#include <string.h> /* for memset */
#include "nrt.h"
#include "assert.h"


typedef int (*atomic_meminfo_cas_func)(MemInfo * volatile *ptr, MemInfo *cmp,
                                       MemInfo *repl, MemInfo **oldptr);


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


struct MemSys{
    /* Ununsed MemInfo are recycled here */
    MemInfo * volatile mi_freelist;
    /* MemInfo with deferred dtor */
    MemInfo * volatile mi_deferlist;
    /* Atomic increment and decrement function */
    atomic_inc_dec_func atomic_inc, atomic_dec;
    /* Atomic CAS */
    atomic_meminfo_cas_func atomic_cas;
    /* Shutdown flag */
    int shutting;

};

/* The Memory System object */
static MemSys TheMSys;


static
MemInfo *nrt_pop_meminfo_list(MemInfo * volatile *list) {
    MemInfo *old, *repl, *head;

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
    } while ( !TheMSys.atomic_cas(list, old, repl, &head));
    return old;
}

static
void nrt_push_meminfo_list(MemInfo * volatile *list, MemInfo *repl) {
    MemInfo *old, *head;
    head = *list;   /* get the current head */
    do {
        old = head; /* old is what CAS compare against */
        /* Set the next item to be the current head */
        repl->list_next = head;
        /* Try to replace the head with the new node.
           The function also perform:
               head <- atomicload(list) */
    } while ( !TheMSys.atomic_cas(list, old, repl, &head) );
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
    if (NULL == newnode) {
        newnode = meminfo_malloc();
    } else {
        assert(newnode->payload.refct == 0 && "RefCt must be 0");
    }
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
    /* Reference count is initialized to zero for easier implementation
       in the compiler.  The compiler incref when value is binding to variables.
       We need to improve the compiler pipeline to better track refcount ops.
     */
    mi->payload.refct = 0;
    mi->payload.dtor = dtor;
    mi->payload.dtor_info = dtor_info;
    mi->payload.data = data;
    mi->payload.size = size;
    return mi;
}

static
void nrt_internal_dtor(void *ptr, void *info) {
    NRT_Debug(nrt_debug_print("nrt_internal_dtor %p, %p\n", ptr, info));
    if (info != NULL) {
        memset(ptr, 0xDE, (size_t)info);  /* for safety */
    }
    NRT_Free(ptr);
}

MemInfo* NRT_MemInfo_alloc(size_t size) {
    void *data = NRT_Allocate(size);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc %p\n", data));
    return NRT_MemInfo_new(data, size, nrt_internal_dtor, NULL);
}

MemInfo* NRT_MemInfo_alloc_safe(size_t size) {
    void *data = NRT_Allocate(size);
    memset(data, 0xCB, size);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_safe %p %llu\n", data, size));
    return NRT_MemInfo_new(data, size, nrt_internal_dtor, (void*)size);
}

void NRT_MemInfo_destroy(MemInfo *mi) {
    NRT_MemSys_insert_meminfo(mi);
}

void NRT_MemInfo_acquire(MemInfo *mi) {
    NRT_Debug(nrt_debug_print("NRT_acquire %p\n", mi));
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
    NRT_Debug(nrt_debug_print("NRT_release %p\n", mi));
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
    return ptr;
}

void NRT_Free(void *ptr) {
    NRT_Debug(nrt_debug_print("NRT_Free %p\n", ptr));
    free(ptr);
}
