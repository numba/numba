#ifndef NUMBAPRO_CPU_SCHEDULER_H
#define NUMBAPRO_CPU_SCHEDULER_H

typedef void (*kernel_t)     (int, int, void* args);
typedef int  (*atomic_add_t) (volatile int *ptr, int val);

struct gang;

struct gang* start_workers(int ncpu, kernel_t kernel, int ntid, void *args,
                      int arglen, atomic_add_t atomic_add);

void join_workers(struct gang *gang);

#endif //NUMBAPRO_CPU_SCHEDULER_H
