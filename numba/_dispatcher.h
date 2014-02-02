#ifndef NUMBA_DISPATCHER_H_
#define NUMBA_DISPATCHER_H_

#ifdef __cplusplus
    extern "C" {
#endif

void*
dispatcher_new(void *tm, int argct);

void
dispatcher_del(void *obj);

void
dispatcher_add_defn(void *obj, int tys[], void* callable);

void*
dispatcher_resolve(void *obj, int sig[], int *matches);

int
dispatcher_count(void *obj);

#ifdef __cplusplus
    }
#endif

#endif  /* NUMBA_DISPATCHER_H_ */
