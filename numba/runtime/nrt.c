#include <stdarg.h>
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
    } payload;

    MemInfo *freelist;
};


struct MemSys{
    MemInfo *mi_freelist;
    atomic_inc_dec_func atomic_inc, atomic_dec;
};

static MemSys TheMSys;

static
MemInfo* meminfo_malloc() {
    void *p = malloc(sizeof(MemInfo));;
    NRT_Debug("meminfo_malloc %p\n", p);
    return p;
}

void NRT_MemSys_insert_meminfo(MemInfo *newnode) {
    MemInfo *prev = TheMSys.mi_freelist;
    if (NULL == newnode) {
        newnode = meminfo_malloc();
    }
    NRT_Debug("NRT_MemSys_insert_meminfo newnode=%p\n", newnode);
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
    NRT_Debug("NRT_MemSys_pop_meminfo: return %p\n", node);
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
    size_t out = *ptr;
    out += 1;
    *ptr = out;
    return out;
}

static
size_t nrt_testing_atomic_dec(size_t *ptr){
    size_t out = *ptr;
    out -= 1;
    *ptr = out;
    return out;
}

void NRT_MemSys_set_atomic_inc_dec_stub(){
    NRT_MemSys_set_atomic_inc_dec(nrt_testing_atomic_inc,
                                  nrt_testing_atomic_dec);
}


MemInfo* NRT_MemInfo_new(void *data, dtor_function dtor, void *dtor_info) {
    MemInfo * mi = NRT_MemSys_pop_meminfo();
    mi->payload.refct = 0;
    mi->payload.dtor = dtor;
    mi->payload.dtor_info = dtor_info;
    mi->payload.data = data;
    return mi;
}

static
void nrt_internal_dtor(void *ptr, void *info) {
    NRT_Debug("nrt_internal_dtor %p, %p\n", ptr, info);
    NRT_Free(ptr);
}

MemInfo* NRT_MemInfo_alloc(size_t size) {
    void *data = NRT_Allocate(size);
    NRT_Debug("NRT_MemInfo_alloc %p\n", data);
    void *meminfo = NRT_MemInfo_new(data, nrt_internal_dtor, NULL);
    return meminfo;
}

void NRT_MemInfo_destroy(MemInfo *mi) {
    NRT_MemSys_insert_meminfo(mi);
}

void NRT_MemInfo_acquire(MemInfo *mi) {
    TheMSys.atomic_inc(&mi->payload.refct);
}

void NRT_MemInfo_release(MemInfo *mi) {
    if (TheMSys.atomic_dec(&mi->payload.refct) == 0) {
        mi->payload.dtor(mi->payload.data, mi->payload.dtor_info);
        NRT_MemSys_insert_meminfo(mi);
    }
}

void* NRT_MemInfo_data(MemInfo* mi) {
    return mi->payload.data;
}

void* NRT_Allocate(size_t size) {
    void *ptr = malloc(size);
    NRT_Debug("NRT_Allocate bytes=%llu ptr=%p\n", size, ptr);
    return ptr;
}

void NRT_Free(void *ptr) {
    NRT_Debug("NRT_Free %p\n", ptr);
    free(ptr);
}

