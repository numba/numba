#ifndef NUMBAPRO_CPU_SCHEDULER_H
#define NUMBAPRO_CPU_SCHEDULER_H

#ifdef __cplusplus
export "C"{
#endif
/*
 * Type for kernel functions
 *
 * id    --- worker id.
 * begin --- first tid.
 * end   --- last tid + 1.
 * args  --- a pointer to a structure of arguments.
 *           the structure must have the following layout:
 *           { int tid, argtype arg0, argtype1 arg1, ..., argtypeN argN }
 */
typedef void (*kernel_t)     (int id, int begin, int end, void* args);

/*
 * Type for atomic add function.
 *
 * ptr --- pointer to an integer which will be added to
 * val --- the value adding to `ptr`.
 */
typedef int (*atomic_add_t) (volatile int *ptr, int val);

struct gang_struct;

struct gang_struct* start_workers(int ncpu, kernel_t kernel, int ntid,
                                  void *args, int arglen,
                                  atomic_add_t atomic_add);

void join_workers(struct gang_struct *gng);


#ifdef __cplusplus
}
#endif

#endif //NUMBAPRO_CPU_SCHEDULER_H
