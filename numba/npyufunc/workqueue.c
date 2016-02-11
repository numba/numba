/*
Implement parallel vectorize workqueue.

This keeps a set of worker threads running all the time.
They wait and spin on a task queue for jobs.

**WARNING**
This module is not thread-safe.  Adding task to queue is not protected from
race condition.
*/

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

static cas_function_t *cas = NULL;

static void
cas_wait(volatile int *ptr, const int old, const int repl) {
    int out = repl;
    int timeout = 1;   /* starting from 1us nap */
    static const int MAX_WAIT_TIME = 20 * 1000; /* max wait is 20ms */

    while (1) {
        if (cas) { /* protect against CAS function being released by LLVM during
                      interpreter teardown. */
            out = cas(ptr, old, repl);
            if (out == old) return;
        }

        take_a_nap(timeout);

        /* Exponentially increase the wait time until the max has reached*/
        timeout <<= 1;
        if (timeout >= MAX_WAIT_TIME) {
            timeout = MAX_WAIT_TIME;
        }
    }
}

/* As the thread-pool isn't inherited by children,
   free the task-queue, too. */
static void reset_after_fork(void);

/* PThread */
#ifdef NUMBA_PTHREAD

static thread_pointer
numba_new_thread(void *worker, void *arg)
{
    int status;
    pthread_attr_t attr;
    pthread_t th;

    pthread_atfork(0, 0, reset_after_fork);

    /* Create detached threads */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    status = pthread_create(&th, &attr, worker, arg);

    if (status != 0){
        return NULL;
    }

    pthread_attr_destroy(&attr);
    return (thread_pointer)th;
}

static void
take_a_nap(int usec) {
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

static unsigned __stdcall
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

static thread_pointer
numba_new_thread(void *worker, void *arg)
{
    uintptr_t handle;
    unsigned threadID;
    callobj *obj;

    if (sizeof(handle) > sizeof(void*))
        return 0;

    obj = (callobj*)HeapAlloc(GetProcessHeap(), 0, sizeof(*obj));
    if (!obj)
        return NULL;

    obj->func = worker;
    obj->arg = arg;

    handle = _beginthreadex(NULL, 0, bootstrap, obj, 0, &threadID);
    if (handle == -1)
        return 0;
    return (thread_pointer)handle;
}

static void
take_a_nap(int usec) {
    /* Note that Sleep(0) will relinquish the current time slice, allowing
       other threads to run. */
    Sleep(usec / 1000);
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

static void
set_cas(void *ptr) {
    cas = ptr;
}

static void
add_task(void *fn, void *args, void *dims, void *steps, void *data) {
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

static void launch_threads(int count) {
    if (!queues) {
        /* If queues are not yet allocated,
           create them, one for each thread. */
       int i;
       size_t sz = sizeof(Queue) * count;

       queues = malloc(sz);     /* this memory will leak */
       memset(queues, 0, sz);
       queue_count = count;

       for (i = 0; i < count; ++i) {
            numba_new_thread(thread_worker, &queues[i]);
       }
    }
}

static void synchronize(void) {
    int i;
    for (i = 0; i < queue_count; ++i) {
        cas_wait(&queues[i].lock, DONE, IDLE);
    }
}

static void ready(void) {
    int i;
    for (i = 0; i < queue_count; ++i) {
        cas_wait(&queues[i].lock, IDLE, READY);
    }
}

static void reset_after_fork(void)
{
    free(queues);
    queues = NULL;
}

MOD_INIT(workqueue) {
    PyObject *m;
    MOD_DEF(m, "workqueue", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

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
