#ifndef NUMBA_DISPATCHER_H_
#define NUMBA_DISPATCHER_H_

#include <Python.h>

#ifdef __cplusplus
    extern "C" {
#endif

typedef struct _opaque_dispatcher dispatcher_t;

dispatcher_t *
dispatcher_new(void *tm, int argct);

void
dispatcher_clear(dispatcher_t *obj);

void
dispatcher_del(dispatcher_t *obj);

void
dispatcher_add_defn(dispatcher_t *obj, int tys[], PyObject* callable);

PyObject*
dispatcher_resolve(dispatcher_t *obj, int sig[], int *matches,
                   int allow_unsafe, int exact_match_required);

int
dispatcher_count(dispatcher_t *obj);

#ifdef __cplusplus
    }
#endif

#endif  /* NUMBA_DISPATCHER_H_ */
