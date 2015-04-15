#include <stdlib.h>
#include <stdio.h>

/* Debugging facilities */
//#undef NDEBUG
#ifndef NDEBUG
#   define NRT_Debug(X) X
#else
#   define NRT_Debug(X)
#endif

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
void NRT_MemSys_init();
void NRT_MemSys_shutdown();
void NRT_MemSys_insert_meminfo(MemInfo *newnode);
MemInfo* NRT_MemSys_pop_meminfo();
void NRT_MemSys_set_atomic_inc_dec(atomic_inc_dec_func inc,
                                   atomic_inc_dec_func dec);
void NRT_MemSys_set_atomic_cas(atomic_cas_func cas);
void NRT_MemSys_set_atomic_inc_dec_stub();
void NRT_MemSys_set_atomic_cas_stub();
void NRT_MemSys_process_defer_dtor();

/* Memory Info API */

/* Create a new MemInfo
 *
 * data: data pointer being tracked
 * dtor: destructor to execute
 * dtor_info: additional information to pass to the destructor
 */
MemInfo* NRT_MemInfo_new(void *data, size_t size, dtor_function dtor,
                         void *dtor_info);
MemInfo* NRT_MemInfo_alloc(size_t size);
MemInfo* NRT_MemInfo_alloc_safe(size_t size);
void NRT_MemInfo_destroy(MemInfo *mi);
void NRT_MemInfo_acquire(MemInfo* mi);
void NRT_MemInfo_release(MemInfo* mi, int defer);
void NRT_MemInfo_call_dtor(MemInfo *mi, int defer);
void* NRT_MemInfo_data(MemInfo* mi);
size_t NRT_MemInfo_size(MemInfo* mi);
void NRT_MemInfo_defer_dtor(MemInfo* mi);

/* General allocator */
void* NRT_Allocate(size_t size);
void NRT_Free(void *ptr);
