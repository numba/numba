#include <stdarg.h>
#include <string.h> /* for memset */
#include "nrt.h"


void nrt_debug_print(char *fmt, ...) {

   va_list args;

   va_start(args, fmt);
   vfprintf(stderr, fmt, args);
   va_end(args);
}

union MemInfo{
    struct {
        size_t         refct;
        dtor_function  dtor;
        void          *dtor_info;
        void          *data;
        size_t         size;
    } payload;

    MemInfo *freelist;
};


struct MemSys{
    /* Ununsed MemInfo are recycled here */
    MemInfo *mi_freelist;
    /* MemInfo with deferred dtor */
    MemInfo *mi_deferlist;
    /* Atomic increment and decrement function */
    atomic_inc_dec_func atomic_inc, atomic_dec;

};

/* The Memory System object */
static MemSys TheMSys;

static
void nrt_meminfo_call_dtor(MemInfo *mi) {
    NRT_Debug(nrt_debug_print("nrt_meminfo_call_dtor\n"));
    /* call dtor */
    mi->payload.dtor(mi->payload.data, mi->payload.dtor_info);
    /* Clear and release MemInfo */
    NRT_MemSys_insert_meminfo(mi);
}

static
MemInfo* meminfo_malloc() {
    void *p = malloc(sizeof(MemInfo));;
    NRT_Debug(nrt_debug_print("meminfo_malloc %p\n", p));
    return p;
}

void NRT_MemSys_init() {
    memset(&TheMSys, 0, sizeof(MemSys));
}

void NRT_MemSys_process_defer_dtor() {
    while (TheMSys.mi_deferlist) {
        /* Pop one */
        MemInfo *mi = TheMSys.mi_deferlist;
        TheMSys.mi_deferlist = mi->freelist;
        NRT_Debug(nrt_debug_print("Defer dtor %p\n", mi));
        nrt_meminfo_call_dtor(mi);
    }
}

void NRT_MemSys_insert_meminfo(MemInfo *newnode) {
    MemInfo *prev = TheMSys.mi_freelist;
    if (NULL == newnode) {
        newnode = meminfo_malloc();
    }
    NRT_Debug(nrt_debug_print("NRT_MemSys_insert_meminfo newnode=%p\n",
                              newnode));
    memset(newnode, 0, sizeof(MemInfo));
    TheMSys.mi_freelist = newnode;
    newnode->freelist = prev;
}

MemInfo* NRT_MemSys_pop_meminfo() {
    MemInfo *node;

    if (NULL == TheMSys.mi_freelist) {
        node = meminfo_malloc();
    } else {
        node = TheMSys.mi_freelist;
        TheMSys.mi_freelist = node->freelist;
    }
    NRT_Debug(nrt_debug_print("NRT_MemSys_pop_meminfo: return %p\n", node));
    return node;
}

void NRT_MemSys_set_atomic_inc_dec(atomic_inc_dec_func inc,
                                   atomic_inc_dec_func dec)
{
    TheMSys.atomic_inc = inc;
    TheMSys.atomic_dec = dec;
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

void NRT_MemSys_set_atomic_inc_dec_stub(){
    NRT_MemSys_set_atomic_inc_dec(nrt_testing_atomic_inc,
                                  nrt_testing_atomic_dec);
}


MemInfo* NRT_MemInfo_new(void *data, size_t size, dtor_function dtor,
                         void *dtor_info)
{
    MemInfo * mi = NRT_MemSys_pop_meminfo();
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
        memset(ptr, 0, (size_t)info);  /* for safety */
    }
    NRT_Free(ptr);
}

MemInfo* NRT_MemInfo_alloc(size_t size) {
    void *data = NRT_Allocate(size);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc %p\n", data));
    void *meminfo = NRT_MemInfo_new(data, size, nrt_internal_dtor, NULL);
    return meminfo;
}

MemInfo* NRT_MemInfo_alloc_safe(size_t size) {
    void *data = NRT_Allocate(size);
    memset(data, 0, size);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_safe %p\n", data));
    void *meminfo = NRT_MemInfo_new(data, size, nrt_internal_dtor, (void*)size);
    return meminfo;
}

void NRT_MemInfo_destroy(MemInfo *mi) {
    NRT_MemSys_insert_meminfo(mi);
}

void NRT_MemInfo_acquire(MemInfo *mi) {
    TheMSys.atomic_inc(&mi->payload.refct);
}

void NRT_MemInfo_release(MemInfo *mi, int defer) {
    /* RefCt drop to zero */
    if (TheMSys.atomic_dec(&mi->payload.refct) == 0) {
        /* We have a destructor */
        if (mi->payload.dtor) {
            if (defer) {
                NRT_MemInfo_defer_dtor(mi);
            } else {
                nrt_meminfo_call_dtor(mi);
            }
        }
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
    mi->freelist = TheMSys.mi_deferlist;
    TheMSys.mi_deferlist = mi;
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

