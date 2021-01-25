/*
Implement parallel vectorize workqueue.

This keeps a set of worker threads running all the time.
They wait and spin on a task queue for jobs.

**WARNING**
This module is not thread-safe.  Adding task to queue is not protected from
race conditions.
*/
#include "../../_pymodule.h"
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#ifdef _MSC_VER
/* Windows */
#include <windows.h>
#include <process.h>
#include <malloc.h>
#include <signal.h>
#define NUMBA_WINTHREAD
#else
/* PThread */
#include <pthread.h>
#include <unistd.h>

#if defined(__FreeBSD__)
#include <stdlib.h>
#else
#include <alloca.h>
#endif

#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#define NUMBA_PTHREAD
#endif

#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include "workqueue.h"
#include "gufunc_scheduler.h"

#define _DEBUG 0

/* workqueue is not threadsafe, so we use DSO globals to flag and update various
 * states.
 */
/* This variable is the nesting level, it's incremented at the start of each
 * parallel region and decremented at the end, if parallel regions are nested
 * on entry the value == 1 and workqueue will abort (this in preference to just
 * hanging or segfaulting).
 */
static int _nesting_level = 0;

/* As the thread-pool isn't inherited by children,
   free the task-queue, too. */
static void reset_after_fork(void);

/* PThread */
#ifdef NUMBA_PTHREAD

typedef struct
{
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

    if (status != 0)
    {
        return NULL;
    }

    pthread_attr_destroy(&attr);
    return (thread_pointer)th;
}

#endif

/* Win Thread */
#ifdef NUMBA_WINTHREAD

typedef struct
{
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
typedef struct
{
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

static int
get_thread_id(void)
{
    return GetCurrentThreadId();
}

#endif

typedef struct Task
{
    void (*func)(void *args, void *dims, void *steps, void *data);
    void *args, *dims, *steps, *data;
    bool broadcast;
} Task;

typedef struct
{
    queue_condition_t cond;
    int state;
    Task task;
    size_t thread_id; // This is a "synthetic" thread id
} Queue;


static Queue *queues = NULL;
static int queue_count;
static int queue_pivot = 0;
static int NUM_THREADS = -1;

static void
queue_state_wait(Queue *queue, int old, int repl)
{
    queue_condition_t *cond = &queue->cond;

    queue_condition_lock(cond);
    while (queue->state != old)
    {
        queue_condition_wait(cond);
    }
    queue->state = repl;
    queue_condition_signal(cond);
    queue_condition_unlock(cond);
}

// break on this for debug
void debug_marker(void);
void debug_marker() {};


#ifdef _MSC_VER
#define THREAD_LOCAL(ty) __declspec(thread) ty
#else
/* Non-standard C99 extension that's understood by gcc and clang */
#define THREAD_LOCAL(ty) __thread ty
#endif

// This is the number of threads that is default, it is set on initialisation of
// the threading backend via the launch_threads() call
static int _INIT_NUM_THREADS = -1;
static int _INIT_THREAD_ID = 0;

// This is the per-thread thread mask, each thread can carry its own mask.
static THREAD_LOCAL(int) _TLS_num_threads = 0;

// This is the per-thread id
static THREAD_LOCAL(int) _TLS_thread_id = -1;

static void
set_num_threads(int count)
{
    _TLS_num_threads = count;
}

static int
get_num_threads(void)
{
    // This is purely to permit the implementation to survive to the point
    // where it can exit cleanly as multiple threads cannot be used with this
    // backend
    if (_TLS_num_threads == 0)
    {
        // This is a thread that did not call launch_threads() but is still a
        // "main" thread, probably from e.g. threading.Thread() use, it still
        // has a TLS slot which is 0 from the lack of launch_threads() call
        _TLS_num_threads = _INIT_NUM_THREADS;
    }
    return _TLS_num_threads;
}

static int
get_thread_id(void)
{
    if (_TLS_thread_id == -1)
    {
        _TLS_thread_id = _INIT_THREAD_ID;
    }
    return (int)_TLS_thread_id;
}


// this complies to a launchable function from `add_task` like:
// add_task(nopfn, NULL, NULL, NULL, NULL)
// useful if you want to limit the number of threads locally
// static void nopfn(void *args, void *dims, void *steps, void *data) {};


// synchronize the TLS num_threads slot to value args[0]
static void sync_num_threads_tls(void *args, void *dims, void *steps, void *data) {
    int nthreads = *((int *)(args));
    _TLS_num_threads = nthreads;
};

// assign a thread_id value based on the queue pivot to the TLS slot thread_id
static void assign_thread_id_tls(void *args, void *dims, void *steps, void *data) {
    int idx = *((int *)(args));
    Queue *queue = &queues[idx];
    _TLS_thread_id = queue->thread_id;
};

// synchronize the TLS sched slot to value args[0]
static void sync_sched_tls(void *args, void *dims, void *steps, void *data) {
    set_sched(args);
};

// synchronize the TLS sched_size slot to value args[0]
static void sync_sched_size(void *args, void *dims, void *steps, void *data) {
    int sz = *((int *)(args));
    set_sched_size(sz);
};

static void
parallel_for(void *fn, char **args, size_t *dimensions, size_t *steps, void *data,
             size_t inner_ndim, size_t array_count, int num_threads)
{

    //     args = <ir.Argument '.1' of type i8**>,
    //     dimensions = <ir.Argument '.2' of type i64*>
    //     steps = <ir.Argument '.3' of type i64*>
    //     data = <ir.Argument '.4' of type i8*>

    // check the nesting level, if it's already 1, abort, workqueue cannot
    // handle nesting.
    if (_nesting_level >= 1){
        fprintf(stderr, "%s", "Terminating: Nested parallel kernel launch "
                              "detected, the workqueue threading layer does "
                              "not supported nested parallelism. Try the TBB "
                              "threading layer.\n");
        raise(SIGABRT);
        return;
    }

    // increment the nest level
    _nesting_level += 1;

    size_t * count_space = NULL;
    char ** array_arg_space = NULL;
    const size_t arg_len = (inner_ndim + 1);
    int i; // induction var for chunking, thread count unlikely to overflow int
    size_t j, count, remain, total;

    ptrdiff_t offset;
    char * base;
    int old_queue_count = -1;

    size_t step;

    debug_marker();

    total = *((size_t *)dimensions);
    count = total / num_threads;
    remain = total;

    if(_DEBUG)
    {
        printf("inner_ndim: %zd\n",inner_ndim);
        printf("arg_len: %zd\n", arg_len);
        printf("total: %zd\n", total);
        printf("count: %zd\n", count);

        printf("dimensions: ");
        for(j = 0; j < arg_len; j++)
        {
            printf("%zd, ", ((size_t *)dimensions)[j]);
        }
        printf("\n");

        printf("steps: ");
        for(j = 0; j < array_count; j++)
        {
            printf("%zd, ", steps[j]);
        }
        printf("\n");

        printf("*args: ");
        for(j = 0; j < array_count; j++)
        {
            printf("%p, ", (void *)args[j]);
        }
    }

    // sync the thread pool TLS slots, sync all slots, we don't know which
    // threads will end up running.
    for (i = 0; i < NUM_THREADS; i++)
    {
        add_task(sync_num_threads_tls, (void *)(&num_threads), NULL, NULL,
                 NULL, true);
    }
    ready();
    synchronize();
    // sync the thread ids
    for (i = 0; i < NUM_THREADS; i++)
    {
        int * buf = (int *)alloca(sizeof(int));
        *buf = i;
        add_task(assign_thread_id_tls, (void *)(buf), NULL, NULL, NULL, true);
    }
    ready();
    synchronize();
    // sync the schedule pointer
    void * sched_ptr = get_sched();
    for (i = 0; i < NUM_THREADS; i++)
    {
        add_task(sync_sched_tls, sched_ptr, NULL, NULL, NULL, true);
    }
    ready();
    synchronize();
    // sync the schedule size
    int sched_size = get_sched_size();
    for (i = 0; i < NUM_THREADS; i++)
    {
        add_task(sync_sched_size, (void *)(&sched_size), NULL, NULL, NULL,
                 true);
    }
    ready();
    synchronize();

    // This backend isn't threadsafe so just mutate the global
    old_queue_count = queue_count;
    queue_count = num_threads;

    for (i = 0; i < num_threads; i++)
    {
        count_space = (size_t *)alloca(sizeof(size_t) * arg_len);
        memcpy(count_space, dimensions, arg_len * sizeof(size_t));
        if(i == num_threads - 1)
        {
            // Last thread takes all leftover
            count_space[0] = remain;
        }
        else
        {
            count_space[0] = count;
            remain = remain - count;
        }

        if(_DEBUG)
        {
            printf("\n=================== THREAD %d ===================\n", i);
            printf("\ncount_space: ");
            for(j = 0; j < arg_len; j++)
            {
                printf("%zd, ", count_space[j]);
            }
            printf("\n");
        }

        array_arg_space = alloca(sizeof(char*) * array_count);

        for(j = 0; j < array_count; j++)
        {
            base = args[j];
            step = steps[j];
            offset = step * count * i;
            array_arg_space[j] = (char *)(base + offset);

            if(_DEBUG)
            {
                printf("Index %zd\n", j);
                printf("-->Got base %p\n", (void *)base);
                printf("-->Got step %zd\n", step);
                printf("-->Got offset %td\n", offset);
                printf("-->Got addr %p\n", (void *)array_arg_space[j]);
            }
        }

        if(_DEBUG)
        {
            printf("\narray_arg_space: ");
            for(j = 0; j < array_count; j++)
            {
                printf("%p, ", (void *)array_arg_space[j]);
            }
        }
        add_task(fn, (void *)array_arg_space, (void *)count_space, steps, data,
                 false);
    }

    ready();
    synchronize();

    queue_count = old_queue_count;
    // decrement the nest level
    _nesting_level -= 1;
}

static void
add_task(void *fn, void *args, void *dims, void *steps, void *data,
         bool broadcast)
{
    void (*func)(void *args, void *dims, void *steps, void *data) = fn;

    Queue *queue = &queues[queue_pivot];

    Task *task = &queue->task;
    task->func = func;
    task->args = args;
    task->dims = dims;
    task->steps = steps;
    task->data = data;
    task->broadcast = broadcast;

    /* Move pivot */
    if ( ++queue_pivot == queue_count )
    {
        queue_pivot = 0;
    }
}

static
void thread_worker(void *arg)
{
    Queue *queue = (Queue*)arg;
    Task *task;

    while (1)
    {
        /* Wait for the queue to be in READY state (i.e. for some task
         * to need running), and switch it to RUNNING.
         */
        queue_state_wait(queue, READY, RUNNING);

        // Only run the task if your thread ID is within the mask or it's a
        // broadcast, this is to ensure that max(thread id) < mask in an
        // executing region.
        task = &queue->task;
        if (task->broadcast || (_TLS_thread_id < _TLS_num_threads)) {
            task->func(task->args, task->dims, task->steps, task->data);
        }

        /* Task is done. */
        queue_state_wait(queue, RUNNING, DONE);
    }
}

static void launch_threads(int count)
{
    if (!queues)
    {
        /* If queues are not yet allocated,
           create them, one for each thread. */
        size_t i;
        size_t sz = sizeof(Queue) * count;

        /* set for use in parallel_for */
        NUM_THREADS = count;
        queues = malloc(sz);     /* this memory will leak */
        /* Note this initializes the state to IDLE */
        memset(queues, 0, sz);
        queue_count = count;

        for (i = 0; i < (size_t)count; ++i)
        {
            queue_condition_init(&queues[i].cond);
            numba_new_thread(thread_worker, &queues[i]);
            queues[i].thread_id = i; // id is just queue index
        }

        _INIT_NUM_THREADS = count;
    }
}

static void synchronize(void)
{
    int i;
    for (i = 0; i < queue_count; ++i)
    {
        queue_state_wait(&queues[i], DONE, IDLE);
    }
}

static void ready(void)
{
    int i;
    for (i = 0; i < queue_count; ++i)
    {
        queue_state_wait(&queues[i], IDLE, READY);
    }
}

static void reset_after_fork(void)
{
    free(queues);
    queues = NULL;
    NUM_THREADS = -1;
    _INIT_NUM_THREADS = -1;
    _nesting_level = 0;
}

MOD_INIT(workqueue)
{
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
    PyObject_SetAttrString(m, "parallel_for",
                           PyLong_FromVoidPtr(&parallel_for));
    PyObject_SetAttrString(m, "do_scheduling_signed",
                           PyLong_FromVoidPtr(&do_scheduling_signed));
    PyObject_SetAttrString(m, "do_scheduling_unsigned",
                           PyLong_FromVoidPtr(&do_scheduling_unsigned));
    PyObject_SetAttrString(m, "set_num_threads",
                           PyLong_FromVoidPtr((void*)&set_num_threads));
    PyObject_SetAttrString(m, "get_num_threads",
                           PyLong_FromVoidPtr((void*)&get_num_threads));
    PyObject_SetAttrString(m, "get_thread_id",
                           PyLong_FromVoidPtr((void*)&get_thread_id));
    PyObject_SetAttrString(m, "set_parallel_chunksize",
                           PyLong_FromVoidPtr((void*)&set_parallel_chunksize));
    PyObject_SetAttrString(m, "get_parallel_chunksize",
                           PyLong_FromVoidPtr((void*)&get_parallel_chunksize));
    PyObject_SetAttrString(m, "compute_sched_size",
                           PyLong_FromVoidPtr((void*)&compute_sched_size));
    PyObject_SetAttrString(m, "get_sched_size",
                           PyLong_FromVoidPtr((void*)&get_sched_size));
    PyObject_SetAttrString(m, "get_sched",
                           PyLong_FromVoidPtr((void*)&get_sched));
    return MOD_SUCCESS_VAL(m);
}
