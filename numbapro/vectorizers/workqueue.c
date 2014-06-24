#ifdef _MSC_VER
    /* Windows */
    #include <windows.h>
	#include <process.h>
    #define NUMBA_WINTHREAD
#else
    /* PThread */
    #include <pthread.h>
    #include <unistd.h>
    #define NUMBA_PTHREAD
#endif

#include <string.h>
#include <stdio.h>
#include "workqueue.h"
#include "../_pymodule.h"

enum QUEUE_STATE {
    IDLE = 0, READY, RUNNING, DONE
};

void take_a_nap(int usec);

typedef int cas_function_t(volatile int *ptr, int old, int val);
static cas_function_t *cas = NULL;

void cas_wait(volatile int *ptr, const int old, const int repl) {
    int out = repl;

    while (1) {
        if (cas) {
            out = cas(ptr, old, repl);
            if (out == old) return;
        }
        take_a_nap(1);
    }
}


/* PThread */
#ifdef NUMBA_PTHREAD

thread_pointer numba_new_thread(void *worker, void *arg)
{
    int status;
    pthread_attr_t attr;
    pthread_t th;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    status = pthread_create(&th, &attr, worker, arg);

    if (status != 0){
        return NULL;
    }

    pthread_attr_destroy(&attr);
    return (thread_pointer)th;
}

int numba_join_thread(thread_pointer thread)
{
    int status;
    pthread_t th = (pthread_t)thread;
    status = pthread_join(th, NULL);
    return status == 0;
}

void take_a_nap(int usec) {
    usleep(usec);
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

thread_pointer numba_new_thread(void *worker, void *arg)
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
    return (thread_pointer)handle;
}

int numba_join_thread(thread_pointer thread)
{
    uintptr_t handle = (uintptr_t)thread;
    WaitForSingleObject(handle, INFINITE);
	CloseHandle(handle);
	return 1;
}


#endif

typedef struct Task{
    void (*func)(void *args, void *dims, void *steps, void *data);
    void *args, *dims, *steps, *data;
} Task;

typedef struct {
    volatile int lock;
    Task task;
} Queue;


static Queue *queues = NULL;
static int queue_count;
static int queue_pivot = 0;

void set_cas(void *ptr) {
    cas = ptr;
}

void add_task(void *fn, void *args, void *dims, void *steps, void *data) {
    void (*func)(void *args, void *dims, void *steps, void *data) = fn;

    Queue *queue = &queues[queue_pivot];

    Task *task = &queue->task;
    task->func = func;
    task->args = args;
    task->dims = dims;
    task->steps = steps;
    task->data = data;

    /* Move pivot */
    if ( ++queue_pivot == queue_count ) {
        queue_pivot = 0;
    }

}

static
void thread_worker(void *arg) {
    Queue *queue = (Queue*)arg;
    Task *task;

    while (1) {
        cas_wait(&queue->lock, READY, RUNNING);

        task = &queue->task;
        task->func(task->args, task->dims, task->steps, task->data);

        cas_wait(&queue->lock, RUNNING, DONE);
    }
}

static
void launch_threads(int count) {
    if ( !queues ) {
        /* If queues are not yet allocated,
           create them, one for each thread. */
       int i;
       size_t sz = sizeof(Queue) * count;

       queues = malloc(sz);
       memset(queues, 0, sz);
       queue_count = count;

       for (i = 0; i < count; ++i) {
            numba_new_thread(thread_worker, &queues[i]);
       }
    }
}

/*
static
void lock_queues() {
    lock(&queue_lock);
    queue_pivot = 0;
}

static
void unlock_queues() {
    unlock(&queue_lock);
}
*/


static
void synchronize() {
    int i;
    for (i = 0; i < queue_count; ++i) {
        cas_wait(&queues[i].lock, DONE, IDLE);
    }
}

static
void ready() {
    int i;
    for (i = 0; i < queue_count; ++i) {
        cas_wait(&queues[i].lock, IDLE, READY);
    }
}


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
    PyObject_SetAttrString(m, "set_cas",
                           PyLong_FromVoidPtr(&set_cas));
    PyObject_SetAttrString(m, "launch_threads",
                           PyLong_FromVoidPtr(&launch_threads));
    PyObject_SetAttrString(m, "synchronize",
                           PyLong_FromVoidPtr(&synchronize));
    PyObject_SetAttrString(m, "ready",
                           PyLong_FromVoidPtr(&ready));
    PyObject_SetAttrString(m, "add_task",
                           PyLong_FromVoidPtr(&add_task));

    return MOD_SUCCESS_VAL(m);
}
