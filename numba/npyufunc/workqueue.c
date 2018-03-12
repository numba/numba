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
#include "gufunc_scheduler.h"

/* As the thread-pool isn't inherited by children,
   free the task-queue, too. */
static void reset_after_fork(void);

/* PThread */
#ifdef NUMBA_PTHREAD

typedef struct {
    pthread_cond_t cond;
    pthread_mutex_t mutex;
} queue_condition_t;

static int
queue_condition_init(queue_condition_t *qc)
{
    int r;
    if ((r = pthread_cond_init(&qc->cond, NULL)))
        return r;
    if ((r = pthread_mutex_init(&qc->mutex, NULL)))
        return r;
    return 0;
}

static void
queue_condition_lock(queue_condition_t *qc)
{
    /* XXX errors? */
    pthread_mutex_lock(&qc->mutex);
}

static void
queue_condition_unlock(queue_condition_t *qc)
{
    /* XXX errors? */
    pthread_mutex_unlock(&qc->mutex);
}

static void
queue_condition_signal(queue_condition_t *qc)
{
    /* XXX errors? */
    pthread_cond_signal(&qc->cond);
}

static void
queue_condition_wait(queue_condition_t *qc)
{
    /* XXX errors? */
    pthread_cond_wait(&qc->cond, &qc->mutex);
}

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

#endif

/* Win Thread */
#ifdef NUMBA_WINTHREAD

typedef struct {
    CONDITION_VARIABLE cv;
    CRITICAL_SECTION cs;
} queue_condition_t;

static int
queue_condition_init(queue_condition_t *qc)
{
    InitializeConditionVariable(&qc->cv);
    InitializeCriticalSection(&qc->cs);
    return 0;
}

static void
queue_condition_lock(queue_condition_t *qc)
{
    EnterCriticalSection(&qc->cs);
}

static void
queue_condition_unlock(queue_condition_t *qc)
{
    LeaveCriticalSection(&qc->cs);
}

static void
queue_condition_signal(queue_condition_t *qc)
{
    WakeConditionVariable(&qc->cv);
}

static void
queue_condition_wait(queue_condition_t *qc)
{
    SleepConditionVariableCS(&qc->cv, &qc->cs, INFINITE);
}

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

#endif

typedef struct Task{
    void (*func)(void *args, void *dims, void *steps, void *data);
    void *args, *dims, *steps, *data;
} Task;

typedef struct {
    queue_condition_t cond;
    int state;
    Task task;
} Queue;


static Queue *queues = NULL;
static int queue_count;
static int queue_pivot = 0;

static void
queue_state_wait(Queue *queue, int old, int repl)
{
    queue_condition_t *cond = &queue->cond;

    queue_condition_lock(cond);
    while (queue->state != old) {
        queue_condition_wait(cond);
    }
    queue->state = repl;
    queue_condition_signal(cond);
    queue_condition_unlock(cond);
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
        /* Wait for the queue to be in READY state (i.e. for some task
         * to need running), and switch it to RUNNING.
         */
        queue_state_wait(queue, READY, RUNNING);

        task = &queue->task;
        task->func(task->args, task->dims, task->steps, task->data);

        /* Task is done. */
        queue_state_wait(queue, RUNNING, DONE);
    }
}

static void launch_threads(int count) {
    if (!queues) {
        /* If queues are not yet allocated,
           create them, one for each thread. */
        int i;
        size_t sz = sizeof(Queue) * count;

        queues = malloc(sz);     /* this memory will leak */
        /* Note this initializes the state to IDLE */
        memset(queues, 0, sz);
        queue_count = count;

        for (i = 0; i < count; ++i) {
            queue_condition_init(&queues[i].cond);
            numba_new_thread(thread_worker, &queues[i]);
        }
    }
}

static void synchronize(void) {
    int i;
    for (i = 0; i < queue_count; ++i) {
        queue_state_wait(&queues[i], DONE, IDLE);
    }
}

static void ready(void) {
    int i;
    for (i = 0; i < queue_count; ++i) {
        queue_state_wait(&queues[i], IDLE, READY);
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

    PyObject_SetAttrString(m, "launch_threads",
                           PyLong_FromVoidPtr(&launch_threads));
    PyObject_SetAttrString(m, "synchronize",
                           PyLong_FromVoidPtr(&synchronize));
    PyObject_SetAttrString(m, "ready",
                           PyLong_FromVoidPtr(&ready));
    PyObject_SetAttrString(m, "add_task",
                           PyLong_FromVoidPtr(&add_task));
    PyObject_SetAttrString(m, "do_scheduling_signed",
                           PyLong_FromVoidPtr(&do_scheduling_signed));
    PyObject_SetAttrString(m, "do_scheduling_unsigned",
                           PyLong_FromVoidPtr(&do_scheduling_unsigned));

    return MOD_SUCCESS_VAL(m);
}
