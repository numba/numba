#include <stdarg.h>
#include <string.h> /* for memset */
#include "nrt.h"
#include "assert.h"

#if !defined MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif


typedef int (*atomic_meminfo_cas_func)(void **ptr, void *cmp,
                                       void *repl, void **oldptr);


/* NOTE: if changing the layout, please update numba.core.runtime.atomicops */
struct MemInfo {
    size_t            refct;
    NRT_dtor_function dtor;
    void              *dtor_info;
    void              *data;
    size_t            size;    /* only used for NRT allocated memory */
};


/*
 * Misc helpers.
 */

static void nrt_fatal_error(const char *msg)
{
    fprintf(stderr, "Fatal Numba error: %s\n", msg);
    fflush(stderr); /* it helps in Windows debug build */

#if defined(MS_WINDOWS) && defined(_DEBUG)
    DebugBreak();
#endif
    abort();
}

/*
 * Global resources.
 */

struct MemSys {
    /* Atomic increment and decrement function */
    NRT_atomic_inc_dec_func atomic_inc, atomic_dec;
    /* Atomic CAS */
    atomic_meminfo_cas_func atomic_cas;
    /* Shutdown flag */
    int shutting;
    /* Stats */
    size_t stats_alloc, stats_free, stats_mi_alloc, stats_mi_free;
    /* System allocation functions */
    struct {
        NRT_malloc_func malloc;
        NRT_realloc_func realloc;
        NRT_free_func free;
    } allocator;
};

/* The Memory System object */
static NRT_MemSys TheMSys;


void NRT_MemSys_init(void) {
    memset(&TheMSys, 0, sizeof(NRT_MemSys));
    /* Bind to libc allocator */
    TheMSys.allocator.malloc = malloc;
    TheMSys.allocator.realloc = realloc;
    TheMSys.allocator.free = free;
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

void NRT_MemSys_set_allocator(NRT_malloc_func malloc_func,
                              NRT_realloc_func realloc_func,
                              NRT_free_func free_func)
{
    if ((malloc_func != TheMSys.allocator.malloc ||
         realloc_func != TheMSys.allocator.realloc ||
         free_func != TheMSys.allocator.free) &&
         (TheMSys.stats_alloc != TheMSys.stats_free ||
          TheMSys.stats_mi_alloc != TheMSys.stats_mi_free)) {
        nrt_fatal_error("cannot change allocator while blocks are allocated");
    }
    TheMSys.allocator.malloc = malloc_func;
    TheMSys.allocator.realloc = realloc_func;
    TheMSys.allocator.free = free_func;
}

void NRT_MemSys_set_atomic_inc_dec(NRT_atomic_inc_dec_func inc,
                                   NRT_atomic_inc_dec_func dec)
{
    TheMSys.atomic_inc = inc;
    TheMSys.atomic_dec = dec;
}

void NRT_MemSys_set_atomic_cas(NRT_atomic_cas_func cas) {
    TheMSys.atomic_cas = (atomic_meminfo_cas_func) cas;
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


/*
 * The MemInfo structure.
 */

void NRT_MemInfo_init(NRT_MemInfo *mi,void *data, size_t size,
                      NRT_dtor_function dtor, void *dtor_info)
{
    mi->refct = 1;  /* starts with 1 refct */
    mi->dtor = dtor;
    mi->dtor_info = dtor_info;
    mi->data = data;
    mi->size = size;
    /* Update stats */
    TheMSys.atomic_inc(&TheMSys.stats_mi_alloc);
}

NRT_MemInfo *NRT_MemInfo_new(void *data, size_t size,
                             NRT_dtor_function dtor, void *dtor_info)
{
    NRT_MemInfo *mi = NRT_Allocate(sizeof(NRT_MemInfo));
    NRT_MemInfo_init(mi, data, size, dtor, dtor_info);
    return mi;
}

size_t NRT_MemInfo_refcount(NRT_MemInfo *mi) {
    /* Should never returns 0 for a valid MemInfo */
    if (mi && mi->data)
        return mi->refct;
    else{
        return (size_t)-1;
    }
}

static
void nrt_internal_dtor_safe(void *ptr, size_t size, void *info) {
    NRT_Debug(nrt_debug_print("nrt_internal_dtor_safe %p, %p\n", ptr, info));
    /* See NRT_MemInfo_alloc_safe() */
    memset(ptr, 0xDE, MIN(size, 256));
}

static
void *nrt_allocate_meminfo_and_data(size_t size, NRT_MemInfo **mi_out) {
    NRT_MemInfo *mi;
    char *base = NRT_Allocate(sizeof(NRT_MemInfo) + size);
    mi = (NRT_MemInfo *) base;
    *mi_out = mi;
    return base + sizeof(NRT_MemInfo);
}


static
void nrt_internal_custom_dtor_safe(void *ptr, size_t size, void *info) {
    NRT_dtor_function dtor = info;
    NRT_Debug(nrt_debug_print("nrt_internal_custom_dtor_safe %p, %p\n",
                              ptr, info));
    if (dtor) {
        dtor(ptr, size, NULL);
    }

    nrt_internal_dtor_safe(ptr, size, NULL);
}


NRT_MemInfo *NRT_MemInfo_alloc(size_t size) {
    NRT_MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data(size, &mi);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc %p\n", data));
    NRT_MemInfo_init(mi, data, size, NULL, NULL);
    return mi;
}

NRT_MemInfo *NRT_MemInfo_alloc_safe(size_t size) {
    return NRT_MemInfo_alloc_dtor_safe(size, NULL);
}

NRT_MemInfo* NRT_MemInfo_alloc_dtor_safe(size_t size, NRT_dtor_function dtor) {
    NRT_MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data(size, &mi);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_dtor_safe %p %zu\n", data, size));
    NRT_MemInfo_init(mi, data, size, nrt_internal_custom_dtor_safe, dtor);
    return mi;
}


static
void *nrt_allocate_meminfo_and_data_align(size_t size, unsigned align,
                                          NRT_MemInfo **mi)
{
    size_t offset, intptr, remainder;
    char *base = nrt_allocate_meminfo_and_data(size + 2 * align, mi);
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

NRT_MemInfo *NRT_MemInfo_alloc_aligned(size_t size, unsigned align) {
    NRT_MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data_align(size, align, &mi);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_aligned %p\n", data));
    NRT_MemInfo_init(mi, data, size, NULL, NULL);
    return mi;
}

NRT_MemInfo *NRT_MemInfo_alloc_safe_aligned(size_t size, unsigned align) {
    NRT_MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data_align(size, align, &mi);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_Debug(nrt_debug_print("NRT_MemInfo_alloc_safe_aligned %p %zu\n",
                              data, size));
    NRT_MemInfo_init(mi, data, size, nrt_internal_dtor_safe, (void*)size);
    return mi;
}

void NRT_MemInfo_destroy(NRT_MemInfo *mi) {
    NRT_Free(mi);
    TheMSys.atomic_inc(&TheMSys.stats_mi_free);
}

void NRT_MemInfo_acquire(NRT_MemInfo *mi) {
    NRT_Debug(nrt_debug_print("NRT_MemInfo_acquire %p refct=%zu\n", mi,
                                                            mi->refct));
    assert(mi->refct > 0 && "RefCt cannot be zero");
    TheMSys.atomic_inc(&mi->refct);
}

void NRT_MemInfo_call_dtor(NRT_MemInfo *mi) {
    NRT_Debug(nrt_debug_print("NRT_MemInfo_call_dtor %p\n", mi));
    if (mi->dtor && !TheMSys.shutting)
        /* We have a destructor and the system is not shutting down */
        mi->dtor(mi->data, mi->size, mi->dtor_info);
    /* Clear and release MemInfo */
    NRT_MemInfo_destroy(mi);
}

void NRT_MemInfo_release(NRT_MemInfo *mi) {
    NRT_Debug(nrt_debug_print("NRT_MemInfo_release %p refct=%zu\n", mi,
                                                            mi->refct));
    assert (mi->refct > 0 && "RefCt cannot be 0");
    /* RefCt drop to zero */
    if (TheMSys.atomic_dec(&mi->refct) == 0) {
        NRT_MemInfo_call_dtor(mi);
    }
}

void* NRT_MemInfo_data(NRT_MemInfo* mi) {
    return mi->data;
}

size_t NRT_MemInfo_size(NRT_MemInfo* mi) {
    return mi->size;
}


void NRT_MemInfo_dump(NRT_MemInfo *mi, FILE *out) {
    fprintf(out, "MemInfo %p refcount %zu\n", mi, mi->refct);
}

/*
 * Resizable buffer API.
 */

static void
nrt_varsize_dtor(void *ptr, size_t size, void *info) {
    NRT_Debug(nrt_debug_print("nrt_varsize_dtor %p\n", ptr));
    if (info) {
        /* call element dtor */
        typedef void dtor_fn_t(void *ptr);
        dtor_fn_t *dtor = info;
        dtor(ptr);
    }
    NRT_Free(ptr);
}

NRT_MemInfo *NRT_MemInfo_new_varsize(size_t size)
{
    NRT_MemInfo *mi;
    void *data = NRT_Allocate(size);
    if (data == NULL)
        return NULL;

    mi = NRT_MemInfo_new(data, size, nrt_varsize_dtor, NULL);
    NRT_Debug(nrt_debug_print("NRT_MemInfo_new_varsize size=%zu "
                              "-> meminfo=%p, data=%p\n", size, mi, data));
    return mi;
}

NRT_MemInfo *NRT_MemInfo_new_varsize_dtor(size_t size, NRT_dtor_function dtor) {
    NRT_MemInfo *mi = NRT_MemInfo_new_varsize(size);
    if (mi) {
        mi->dtor_info = dtor;
    }
    return mi;
}

void *NRT_MemInfo_varsize_alloc(NRT_MemInfo *mi, size_t size)
{
    if (mi->dtor != nrt_varsize_dtor) {
        nrt_fatal_error("ERROR: NRT_MemInfo_varsize_alloc called "
                        "with a non varsize-allocated meminfo");
        return NULL;  /* unreachable */
    }
    mi->data = NRT_Allocate(size);
    if (mi->data == NULL)
        return NULL;
    mi->size = size;
    NRT_Debug(nrt_debug_print("NRT_MemInfo_varsize_alloc %p size=%zu "
                              "-> data=%p\n", mi, size, mi->data));
    return mi->data;
}

void *NRT_MemInfo_varsize_realloc(NRT_MemInfo *mi, size_t size)
{
    if (mi->dtor != nrt_varsize_dtor) {
        nrt_fatal_error("ERROR: NRT_MemInfo_varsize_realloc called "
                        "with a non varsize-allocated meminfo");
        return NULL;  /* unreachable */
    }
    mi->data = NRT_Reallocate(mi->data, size);
    if (mi->data == NULL)
        return NULL;
    mi->size = size;
    NRT_Debug(nrt_debug_print("NRT_MemInfo_varsize_realloc %p size=%zu "
                              "-> data=%p\n", mi, size, mi->data));
    return mi->data;
}

void NRT_MemInfo_varsize_free(NRT_MemInfo *mi, void *ptr)
{
    NRT_Free(ptr);
    if (ptr == mi->data)
        mi->data = NULL;
}

/*
 * Low-level allocation wrappers.
 */

void* NRT_Allocate(size_t size) {
    void *ptr = TheMSys.allocator.malloc(size);
    NRT_Debug(nrt_debug_print("NRT_Allocate bytes=%zu ptr=%p\n", size, ptr));
    TheMSys.atomic_inc(&TheMSys.stats_alloc);
    return ptr;
}

void *NRT_Reallocate(void *ptr, size_t size) {
    void *new_ptr = TheMSys.allocator.realloc(ptr, size);
    NRT_Debug(nrt_debug_print("NRT_Reallocate bytes=%zu ptr=%p -> %p\n",
                              size, ptr, new_ptr));
    return new_ptr;
}

void NRT_Free(void *ptr) {
    NRT_Debug(nrt_debug_print("NRT_Free %p\n", ptr));
    TheMSys.allocator.free(ptr);
    TheMSys.atomic_inc(&TheMSys.stats_free);
}

/*
 * Debugging printf function used internally
 */
void nrt_debug_print(char *fmt, ...) {
   va_list args;

   va_start(args, fmt);
   vfprintf(stderr, fmt, args);
   va_end(args);
}


static
void nrt_manage_memory_dtor(void *data, size_t size, void *info) {
    NRT_managed_dtor* dtor = (NRT_managed_dtor*)info;
    dtor(data);
}

static
NRT_MemInfo* nrt_manage_memory(void *data, NRT_managed_dtor dtor) {
    return NRT_MemInfo_new(data, 0, nrt_manage_memory_dtor, dtor);
}


static const
NRT_api_functions nrt_functions_table = {
    NRT_MemInfo_alloc,
    nrt_manage_memory,
    NRT_MemInfo_acquire,
    NRT_MemInfo_release,
    NRT_MemInfo_data
};


const NRT_api_functions* NRT_get_api(void) {
    return &nrt_functions_table;
}
