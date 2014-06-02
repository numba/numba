#ifdef _MSC_VER
    /* Windows */
    #include <windows.h>
	#include <process.h>
    #define NUMBA_WINTHREAD
#else
    /* PThread */
    #include <pthread.h>
    #define NUMBA_PTHREAD
#endif

#include <stdio.h>
#include "workqueue.h"
#include "../_pymodule.h"

/* PThread */
#ifdef NUMBA_PTHREAD

void* numba_new_thread(void *worker, void *arg)
{
    int status;
    pthread_t th;
    status = pthread_create(&th, NULL, worker, arg);
    if (status != 0){
        return NULL;
    }
    return th;
}

int numba_join_thread(void *thread)
{
    int status;
    pthread_t th = thread;
    status = pthread_join(th, NULL);
    return status == 0;
}

#endif

/* Win Thread */
#ifdef NUMBA_WINTHREAD

/* Adapted from Python/thread_nt.h */
typedef struct {
    void (*func)(void*);
    void *arg;
} callobj;

static
unsigned __stdcall
bootstrap(void *call)
{
    callobj *obj = (callobj*)call;
    void (*func)(void*) = obj->func;
    void *arg = obj->arg;
    HeapFree(GetProcessHeap(), 0, obj);
    func(arg);
	_endthreadex(0);
    return 0;
}

void* numba_new_thread(void *worker, void *arg)
{
    uintptr_t handle;
    unsigned threadID;
    callobj *obj;

	if (sizeof(handle) > sizeof(void*)) return 0;

    obj = (callobj*)HeapAlloc(GetProcessHeap(), 0, sizeof(*obj));
    if (!obj)
        return NULL;

    obj->func = worker;
    obj->arg = arg;

    handle = _beginthreadex(NULL, 0, bootstrap, obj, 0, &threadID);
    if (handle == -1) return 0;
    return (void*)handle;
}

int numba_join_thread(void *thread)
{
    uintptr_t handle = thread;
    WaitForSingleObject(handle, INFINITE);
	CloseHandle(handle);
	return 1;
}


#endif

/*MARK1*/

MOD_INIT(workqueue) {
    PyObject *m;
    MOD_DEF(m, "workqueue", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    /*MARK2*/
    PyObject_SetAttrString(m, "new_thread_fnptr",
                           PyLong_FromVoidPtr(&numba_new_thread));
    PyObject_SetAttrString(m, "join_thread_fnptr",
                           PyLong_FromVoidPtr(&numba_join_thread));

    return MOD_SUCCESS_VAL(m);
}
