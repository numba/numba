#ifndef PARALLEL_DISPATCH_H_
#define PARALLEL_DISPATCH_H_

#include <stdio.h>
#include <stdint.h>
#include <pthread.h>

#define printf(...) //disable printf

typedef struct {
    volatile npy_intp next, last;
    volatile int lock;
} ParallelWorkQueue;

typedef struct {
    // loop ufunc args
    char **args;
    npy_intp *dimensions;
    npy_intp *steps;
    void *data;

    // specifics for parallel work queues
    void * func;
    unsigned num_thread;
    ParallelWorkQueue *workqueues; // length at least num_thread
} ParallelContextCommon;

typedef struct {
    ParallelContextCommon *common;
    unsigned id;
} ParallelContext;

static void atomic_lock_workqueue(ParallelWorkQueue * queue){
    while ( !__sync_bool_compare_and_swap(&queue->lock, 0, 1) );
}

static void atomic_unlock_workqueue(ParallelWorkQueue * queue){
    if ( !__sync_bool_compare_and_swap(&queue->lock, 1, 0) )
        exit(1); // Kill on error.
}

static inline
void ufunc_worker_core_d_d(ParallelContextCommon * C, npy_intp i)
{
    typedef double (function_t)(double);
    function_t *func = C->func;
    char * const in_base = C->args[0], * const out_base = C->args[1];
    npy_intp in_step = C->steps[0], out_step = C->steps[1];

    // real work
    double * in = (double*)(in_base + (i * in_step));
    double * out = (double*)(out_base + (i * out_step));

    *out = func(*in);
}


static inline
void ufunc_worker_core_i_b(ParallelContextCommon * C, npy_intp i)
{
    typedef char (function_t)(int);
    function_t *func = C->func;
    char * const in_base = C->args[0], * const out_base = C->args[1];
    npy_intp in_step = C->steps[0], out_step = C->steps[1];


    // real work
     int * in = (int*)(in_base + (i * in_step));
     char * out = (char*)(out_base + (i * out_step));

     *out = func(*in);
}



static
void ufunc_worker_d_d(ParallelContext * context)
{
    ParallelContextCommon *C = context->common;
    const unsigned tid = context->id;

    ParallelWorkQueue *queue = &C->workqueues[tid];

    while (1){
        atomic_lock_workqueue(queue);

        const npy_intp item = queue->next++;
        const npy_intp Last = queue->last;

        atomic_unlock_workqueue(queue);

        if ( item >= Last ) {
            break;
        }

        ufunc_worker_core_d_d(C, item);
    }

    // Do work stealing from other threads.
    int at_least_one_thread_has_not_completed;
    do {
        at_least_one_thread_has_not_completed = 0;
        unsigned i;
        for (i = 0; i < C->num_thread; ++i) {
            if (i != tid) {
                ParallelWorkQueue * otherqueue = &C->workqueues[i];

                atomic_lock_workqueue(otherqueue);

                if (otherqueue->next < otherqueue->last){

                    const npy_intp item = --otherqueue->last;

                    atomic_unlock_workqueue(otherqueue);

                    ufunc_worker_core_d_d(C, item);

                    at_least_one_thread_has_not_completed = 1;
                } else {
                    atomic_unlock_workqueue(otherqueue);
                }
            }
        }
    } while(at_least_one_thread_has_not_completed);
}



static
void ufunc_worker_i_b(ParallelContext * context)
{
    ParallelContextCommon *C = context->common;
    const unsigned tid = context->id;

    ParallelWorkQueue *queue = &C->workqueues[tid];

    while (1){
        atomic_lock_workqueue(queue);

        const npy_intp item = queue->next++;
        const npy_intp Last = queue->last;

        atomic_unlock_workqueue(queue);

        if ( item >= Last ) {
            break;
        }

        ufunc_worker_core_i_b(C, item);
    }

    // Do work stealing from other threads.
    int at_least_one_thread_has_not_completed;
    do {
        at_least_one_thread_has_not_completed = 0;
        unsigned i;
        for (i = 0; i < C->num_thread; ++i) {
            if (i != tid) {
                ParallelWorkQueue * otherqueue = &C->workqueues[i];

                atomic_lock_workqueue(otherqueue);

                if (otherqueue->next < otherqueue->last){

                    const npy_intp item = --otherqueue->last;

                    atomic_unlock_workqueue(otherqueue);

                    ufunc_worker_core_i_b(C, item);

                    at_least_one_thread_has_not_completed = 1;
                } else {
                    atomic_unlock_workqueue(otherqueue);
                }
            }
        }
    } while(at_least_one_thread_has_not_completed);
}

static
void parallel_ufunc(void * func, void * worker,
                        char **args, npy_intp *dimensions,
                        npy_intp *steps, void *data)
{
    enum { NUM_THREAD = 2 };
    ParallelContextCommon C;
    ParallelWorkQueue workqueues[NUM_THREAD];
    // prepare first context
    C.args = args;
    C.dimensions = dimensions;
    C.steps = steps;
    C.data = data;

    C.func = func;

    C.num_thread = NUM_THREAD;

    C.workqueues = workqueues;

    const npy_intp N = dimensions[0];
    const npy_intp ChunkSize = N / NUM_THREAD;

    unsigned i;

    for (i = 0; i < NUM_THREAD; ++i){
        workqueues[i].next = i * ChunkSize;
        workqueues[i].last = (i + 1) * ChunkSize;
        workqueues[i].lock = 0;
    }
    // ensure last one reaches the end
    workqueues[NUM_THREAD-1].last = N;

    // prepare context
    ParallelContext contexts[NUM_THREAD];

    for (i=0; i < NUM_THREAD; ++i){
        contexts[i].common = &C;
        contexts[i].id = i;
    }

    // launch
    pthread_t threads[NUM_THREAD];
    for (i = 0; i < NUM_THREAD; ++i ){
        pthread_create(&threads[i], NULL, worker, (void*)&contexts[i]);
    }

    // join
    for (i = 0; i < NUM_THREAD; ++i ){
        pthread_join(threads[i], NULL);
    }

}

#endif  // PARALLEL_DISPATCH_H_
