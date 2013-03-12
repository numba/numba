#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cpuscheduler.h"
#define MAX_BURST 128

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct queue {
    volatile int begin, end;
    volatile int lock;
} queue_t;

typedef struct worker {
    int             id;
    queue_t         queue;
    struct gang    *parent;
} worker_t;

#if defined(__WIN32__) || defined(_WIN32) || defined(_MSC_VER)
#include <windows.h>
#define STATIC_INLINE static
typedef struct winthread_worker {
    worker_t base;
    HANDLE   thread;
} winthread_worker_t;
#else
#include <pthread.h>
#define STATIC_INLINE static inline
typedef struct pthread_worker {
    worker_t    base;
    pthread_t   thread;
} pthread_worker_t;
#endif

typedef struct gang{
    worker_t    **members;
    int           len;
    // Count completed threads which moved into workstealing mode.
    // Only used when workstealing is enabled.
    volatile int  done;
    // Counter for kernel execution; used to check race condition.
    // Only used when workstealing is enabled.
    volatile int  runct;
    int           ntid;
    kernel_t      kernel;
    void         *args;
    // Atomic add function.  Can be NULL, which means disable workstealing.
    atomic_add_t  atomic_add;
} gang_t;


// globals


STATIC_INLINE
void* malloc_zero(unsigned sz){
    void* p = malloc(sz);
    memset(p, 0, sz);
    return p;
}

STATIC_INLINE
void lock(gang_t* gng, volatile int *ptr){
    int old;
    while (1){
        old = gng->atomic_add(ptr, 1);
        if (old == 0) return;
        else gng->atomic_add(ptr, -1); // undo
    }
}

STATIC_INLINE
void unlock(gang_t* gng, volatile int *ptr){
    gng->atomic_add(ptr, -1);
}

STATIC_INLINE
gang_t* init_gang(int ncpu, kernel_t kernel, int ntid,
                  void *args, int arglen,
                  atomic_add_t atomic_add,
                  int *taskperqueue)
{
    *taskperqueue = ntid / ncpu + (ntid % ncpu ? 1 : 0);

    gang_t *gng  = malloc_zero(sizeof(gang_t));
    gng->len = ncpu;
    gng->done = 0;
    gng->members = malloc_zero(ncpu * sizeof(void*));
    gng->args = malloc_zero(arglen);
    gng->kernel = kernel;
    gng->ntid = ntid;
    gng->runct = 0;
    gng->atomic_add = atomic_add;

    memcpy(gng->args, args, arglen); // keep copy of the arguments

    return gng;
}

STATIC_INLINE
void init_workers(gang_t *gng, int sizeof_worker, int taskperqueue)
{
    int i;
    int ntask = 0;
    for (i = 0; i < gng->len; ++i) {
        worker_t *worker = malloc_zero(sizeof(pthread_worker_t));
        int begin = ntask;
        ntask += taskperqueue;
        int end = MIN(ntask, gng->ntid);

        worker->id = i;
        worker->queue.begin = begin;
        worker->queue.end   = end;
        worker->queue.lock  = 0;
        worker->parent      = gng;

        gng->members[i] = (worker_t*)worker;
    }
}

STATIC_INLINE
gang_t* init_gang_workers(int ncpu, kernel_t kernel, int ntid,
                          void *args, int arglen, atomic_add_t atomic_add,
                          int sizeof_worker)
{
    int taskperqueue;
    gang_t* gng = init_gang(ncpu, kernel, ntid, args, arglen, atomic_add,
                             &taskperqueue);
    init_workers(gng, sizeof_worker, taskperqueue);
    return gng;
}

STATIC_INLINE
void fini_gang(gang_t *gng)
{
    if (gng->atomic_add && gng->ntid != gng->runct) {
        printf("race condition detected: ntid=%d runct=%d\n",
               gng->ntid, gng->runct);
        exit(1);
    }
    free(gng->args);
    free(gng->members);
    free(gng);
}

static
void run_worker(worker_t *worker)
{
    if (worker->parent->atomic_add){
        // When workstealing is enabled, each worker split the queue into
        // smaller burst and loops until all work is done.
        while (1){
            // lock
            lock(worker->parent, &worker->queue.lock);
            // critical section
            int begin = worker->queue.begin;
            int end = MIN(begin + MAX_BURST, worker->queue.end);
            worker->queue.begin += end - begin;
            // unlock
            unlock(worker->parent, &worker->queue.lock);
            if (begin >= end) break;
            // run kernel
            worker->parent->kernel(worker->id, begin, end,
                                   worker->parent->args);
            worker->parent->atomic_add(&worker->parent->runct, end - begin);
        }
        worker->parent->atomic_add(&worker->parent->done, 1);
    } else {
        // When workstealing is disabled, each worker simply run on the entire
        // range of the queue in one go.
        int begin = worker->queue.begin;
        int end = worker->queue.end;
        // run kernel
        worker->parent->kernel(worker->id, begin, end, worker->parent->args);
    }

    if (!worker->parent->atomic_add) {
        // workstealing is diabled
        return;
    }
    // I'm done with my work. Let's steal some from others.
    worker_t **peers = worker->parent->members;
    const int npeer = worker->parent->len;
    worker_t * peer;
    while (worker->parent->done < worker->parent->len) {
        // find a peer that I can steal from
        int ip;
        for(ip = 0; ip < npeer; ++ip){
            if (peers[ip] != worker) {   // not self
                peer = peers[ip];
                if (peer->queue.begin + MAX_BURST < peer->queue.end){
                    break; // found candidate
                }
            }
        }
        
        // lock
        lock(worker->parent, &peer->queue.lock);
        // critical section
        int end = peer->queue.end;
        int begin = MAX(peer->queue.begin, end - MAX_BURST);
        peer->queue.end -= end - begin;
        //unlock
        unlock(worker->parent, &peer->queue.lock);
        if (begin < end) {
            // run kernel
            worker->parent->kernel(worker->id, begin, end, worker->parent->args);
            worker->parent->atomic_add(&worker->parent->runct, end - begin);
        }
    }
}
//#define NOTHREAD
#ifdef NOTHREAD

gang_t* start_workers(int ncpu, kernel_t kernel, int ntid, void *args,
                      int arglen, atomic_add_t ignored){
    kernel(0, ntid, args);
    return NULL;
}


void join_workers(gang_t *gng){ }


#else

#if defined(__WIN32__) || defined(_WIN32) || defined(_MSC_VER)
gang_t* start_workers(int ncpu, kernel_t kernel, int ntid, void *args,
                      int arglen, atomic_add_t atomic_add)
{
    int i;
    gang_t *gng = init_gang_workers(ncpu, kernel, ntid, args, arglen,
                                     atomic_add, sizeof(winthread_worker_t));

    for (i = 0; i < gng->len; ++i) {
        winthread_worker_t* worker = (winthread_worker_t*)gng->members[i];
        worker->thread = CreateThread(NULL, 0, (void*)run_worker,
                                      (void*)&worker->base, 0, NULL);
    }
    return gng;
}


void join_workers(gang_t *gng)
{
    int i;
    for (i = 0; i < gng->len; ++i){
        winthread_worker_t* worker = (winthread_worker_t*)gng->members[i];
        WaitForSingleObject(worker->thread, INFINITE);
        CloseHandle(worker->thread);
        free(worker);
    }
    fini_gang(gng);
}
#else
gang_t* start_workers(int ncpu, kernel_t kernel, int ntid, void *args,
                      int arglen, atomic_add_t atomic_add)
{
    int i;
    gang_t *gng = init_gang_workers(ncpu, kernel, ntid, args, arglen,
                                     atomic_add, sizeof(pthread_worker_t));

    for (i = 0; i < gng->len; ++i) {
        pthread_worker_t* worker = (pthread_worker_t*)gng->members[i];
        pthread_create(&worker->thread, NULL, (void*(*)(void*))run_worker,
                       &worker->base);
    }
    return gng;
}


void join_workers(gang_t *gng)
{
    int i;
    for (i = 0; i < gng->len; ++i){
        pthread_worker_t* worker = (pthread_worker_t*)gng->members[i];
        pthread_join(worker->thread, NULL);
        free(worker);
    }
    fini_gang(gng);
}
#endif

#endif // NOTHREAD

