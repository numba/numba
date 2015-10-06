/*
All functions described here are threadsafe.
*/

#include <stdlib.h>
#include <stdio.h>

/* Debugging facilities - enabled at compile-time */
/* #undef NDEBUG */
#if 0
#   define NRT_Debug(X) X
#else
#   define NRT_Debug(X) if (0) { X; }
#endif

/* TypeDefs */
typedef void (*NRT_dtor_function)(void *ptr, void *info);
typedef size_t (*NRT_atomic_inc_dec_func)(size_t *ptr);
typedef int (*NRT_atomic_cas_func)(void * volatile *ptr, void *cmp, void *repl,
                                   void **oldptr);

typedef struct MemInfo NRT_MemInfo;
typedef struct MemSys NRT_MemSys;

typedef void *(*NRT_malloc_func)(size_t size);
typedef void *(*NRT_realloc_func)(void *ptr, size_t new_size);
typedef void (*NRT_free_func)(void *ptr);


/* Memory System API */

/* Initialize the memory system */
void NRT_MemSys_init(void);

/* Shutdown the memory system */
void NRT_MemSys_shutdown(void);

/*
 * Register the system allocation functions
 */
void NRT_MemSys_set_allocator(NRT_malloc_func, NRT_realloc_func, NRT_free_func);

/*
 * Register the atomic increment and decrement functions
 */
void NRT_MemSys_set_atomic_inc_dec(NRT_atomic_inc_dec_func inc,
                                   NRT_atomic_inc_dec_func dec);


/*
 * Register the atomic compare and swap function
 */
void NRT_MemSys_set_atomic_cas(NRT_atomic_cas_func cas);

/*
 * Register a non-atomic STUB for increment and decrement
 */
void NRT_MemSys_set_atomic_inc_dec_stub(void);

/*
 * Register a non-atomic STUB for compare and swap
 */
void NRT_MemSys_set_atomic_cas_stub(void);

/*
 * The following functions get internal statistics of the memory subsystem.
 */
size_t NRT_MemSys_get_stats_alloc(void);
size_t NRT_MemSys_get_stats_free(void);
size_t NRT_MemSys_get_stats_mi_alloc(void);
size_t NRT_MemSys_get_stats_mi_free(void);

/* Memory Info API */

/* Create a new MemInfo for external memory
 *
 * data: data pointer being tracked
 * dtor: destructor to execute
 * dtor_info: additional information to pass to the destructor
 */
NRT_MemInfo* NRT_MemInfo_new(void *data, size_t size,
                             NRT_dtor_function dtor, void *dtor_info);

void NRT_MemInfo_init(NRT_MemInfo *mi, void *data, size_t size,
                      NRT_dtor_function dtor, void *dtor_info);

/*
 * Returns the refcount of a MemInfo or (size_t)-1 if error.
 */
size_t NRT_MemInfo_refcount(NRT_MemInfo *mi);

/*
 * Allocate memory of `size` bytes and return a pointer to a MemInfo structure
 * that describes the allocation
 */
NRT_MemInfo *NRT_MemInfo_alloc(size_t size);

/*
 * The "safe" NRT_MemInfo_alloc performs additional steps to help debug
 * memory errors.
 * It is guaranteed to:
 *   - zero-fill to the memory region after allocation and before deallocation.
 *   - may do more in the future
 */
NRT_MemInfo *NRT_MemInfo_alloc_safe(size_t size);

/*
 * Aligned versions of the NRT_MemInfo_alloc and NRT_MemInfo_alloc_safe.
 * These take an additional argument `align` for number of bytes to align to.
 */
NRT_MemInfo *NRT_MemInfo_alloc_aligned(size_t size, unsigned align);
NRT_MemInfo *NRT_MemInfo_alloc_safe_aligned(size_t size, unsigned align);

/*
 * Internal API.
 * Release a MemInfo. Calls NRT_MemSys_insert_meminfo.
 */
void NRT_MemInfo_destroy(NRT_MemInfo *mi);

/*
 * Acquire a reference to a MemInfo
 */
void NRT_MemInfo_acquire(NRT_MemInfo* mi);

/*
 * Release a reference to a MemInfo
 */
void NRT_MemInfo_release(NRT_MemInfo* mi);

/*
 * Internal/Compiler API.
 * Invoke the registered destructor of a MemInfo.
 */
void NRT_MemInfo_call_dtor(NRT_MemInfo *mi);

/*
 * Returns the data pointer
 */
void* NRT_MemInfo_data(NRT_MemInfo* mi);

/*
 * Returns the allocated size
 */
size_t NRT_MemInfo_size(NRT_MemInfo* mi);


/*
 * NRT API for resizable buffers.
 */

NRT_MemInfo *NRT_MemInfo_varsize_alloc(size_t size);
void *NRT_MemInfo_varsize_realloc(NRT_MemInfo *mi, size_t size);


/*
 * Print debug info to FILE
 */
void NRT_MemInfo_dump(NRT_MemInfo *mi, FILE *out);


/* Low-level allocation wrappers. */

/*
 * Allocate memory of `size` bytes.
 */
void* NRT_Allocate(size_t size);

/*
 * Deallocate memory pointed by `ptr`.
 */
void NRT_Free(void *ptr);

/*
 * Reallocate memory at `ptr`.
 */
void *NRT_Reallocate(void *ptr, size_t size);

/*
 * Debugging printf function used internally
 */
void nrt_debug_print(char *fmt, ...);
