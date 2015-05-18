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
#   define NRT_Debug(X)
#endif


/*
 * Debugging printf function used internally
 */
static
void nrt_debug_print(char *fmt, ...) {

   va_list args;

   va_start(args, fmt);
   vfprintf(stderr, fmt, args);
   va_end(args);
}


/* TypeDefs */
typedef void (*dtor_function)(void *ptr, void *info);
typedef size_t (*atomic_inc_dec_func)(size_t *ptr);
typedef int (*atomic_cas_func)(void * volatile *ptr, void *cmp, void *repl,
                               void **oldptr);

typedef union MemInfo MemInfo;
typedef struct MemSys MemSys;

/* Memory System API */

/* Initialize the memory system */
void NRT_MemSys_init(void);

/* Shutdown the memory system */
void NRT_MemSys_shutdown(void);

/*
 * Internal API
 * Add a new MemInfo to the freelist (available MemInfo for use).
 * If `newnode` is NULL, a new MemInfo is created.
 */
void NRT_MemSys_insert_meminfo(MemInfo *newnode);


/*
 * Internal API
 * Pop a MemInfo from the freelist.
 */
MemInfo* NRT_MemSys_pop_meminfo(void);

/*
 * Register the atomic increment and decrement functions
 */
void NRT_MemSys_set_atomic_inc_dec(atomic_inc_dec_func inc,
                                   atomic_inc_dec_func dec);


/*
 * Register the atomic compare and swap function
 */
void NRT_MemSys_set_atomic_cas(atomic_cas_func cas);

/*
 * Register a non-atomic STUB for increment and decrement
 */
void NRT_MemSys_set_atomic_inc_dec_stub(void);

/*
 * Register a non-atomic STUB for compare and swap
 */
void NRT_MemSys_set_atomic_cas_stub(void);

/*
 * Process all pending deferred dtors
 */
void NRT_MemSys_process_defer_dtor(void);

/* Memory Info API */

/* Create a new MemInfo
 *
 * data: data pointer being tracked
 * dtor: destructor to execute
 * dtor_info: additional information to pass to the destructor
 */
MemInfo* NRT_MemInfo_new(void *data, size_t size, dtor_function dtor,
                         void *dtor_info);

/*
 * Allocate memory of `size` bytes and return a pointer to a MemInfo structure
 * that describes the allocation
 */
MemInfo* NRT_MemInfo_alloc(size_t size);

/*
 * The "safe" NRT_MemInfo_alloc performs additional steps to help debug
 * memory errors.
 * It is guaranteed to:
 *   - zero-fill to the memory region after allocation and before deallocation.
 *   - may do more in the future
 */
MemInfo* NRT_MemInfo_alloc_safe(size_t size);

/*
 * Internal API.
 * Release a MemInfo. Calls NRT_MemSys_insert_meminfo.
 */
void NRT_MemInfo_destroy(MemInfo *mi);

/*
 * Acquire a reference to a MemInfo
 */
void NRT_MemInfo_acquire(MemInfo* mi);

/*
 * Release a reference to a MemInfo
 */
void NRT_MemInfo_release(MemInfo* mi, int defer);

/*
 * Internal/Compiler API.
 * Invoke the registered destructor of a MemInfo.
 * if `defer` is true, the MemInfo is added to the deferred dtor list.
 * Calls NRT_MemInfo_defer_dtor.
 */
void NRT_MemInfo_call_dtor(MemInfo *mi, int defer);

/*
 * Returns the data pointer
 */
void* NRT_MemInfo_data(MemInfo* mi);

/*
 * Returns the allocated size
 */
size_t NRT_MemInfo_size(MemInfo* mi);

/*
 * Internal API.
 * Append the MemInfo to the deferred dtor list.
 */
void NRT_MemInfo_defer_dtor(MemInfo* mi);


/*
 * Print debug info to FILE
 */
void NRT_MemInfo_dump(MemInfo *mi, FILE *out);

/* General allocator */

/*
 * Allocate memory of `size` bytes.
 */
void* NRT_Allocate(size_t size);

/*
 * Deallocate memory pointed by `ptr`.
 */
void NRT_Free(void *ptr);
