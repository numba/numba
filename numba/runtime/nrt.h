
#include <stdlib.h>
#include <stdio.h>

/* Debugging facilities */
void nrt_debug_print(char *fmt, ...);
#define NRT_Debug nrt_debug_print

/* TypeDefs */
typedef void (*dtor_function)(void *ptr, void *info);
typedef size_t (*atomic_inc_dec_func)(size_t *ptr);

typedef union MemInfo MemInfo;
typedef struct MemSys MemSys;

/* Memory System API */
void NRT_MemSys_insert_meminfo(MemInfo *newnode);
MemInfo* NRT_MemSys_pop_meminfo();
void NRT_MemSys_set_atomic_inc_dec(atomic_inc_dec_func inc,
                                   atomic_inc_dec_func dec);
void NRT_MemSys_set_atomic_inc_dec_stub();

/* Memory Info API */
MemInfo* NRT_MemInfo_new(void *data, dtor_function dtor, void *dtor_info);
MemInfo* NRT_MemInfo_alloc(size_t size);
void NRT_MemInfo_destroy(MemInfo *mi);
void NRT_MemInfo_acquire(MemInfo* mi);
void NRT_MemInfo_release(MemInfo* mi);
void* NRT_MemInfo_data(MemInfo* mi);

/* General allocator */
void* NRT_Allocate(size_t size);
void NRT_Free(void *ptr);
