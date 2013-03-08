#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cpuscheduler.h"
#define MAX_BURST 64

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


typedef struct queue {
    volatile int begin, end;
    volatile int lock;
} queue_t;

typedef struct worker {
    queue_t         queue;
    struct gang    *parent;
} worker_t;

#if defined(__WIN32__)
#include <windows.h>
typedef struct winthread_worker {
    worker_t base;
    HANDLE   thread;
} pthread_worker_t;
#else
#include <pthread.h>
typedef struct pthread_worker {
    worker_t    base;
    pthread_t   thread;
} pthread_worker_t;
#endif

typedef struct gang{
    worker_t    **members;
    int           len;
    volatile int  done;
    volatile int  runct;
    int           ntid;
    kernel_t      kernel;
    void         *args;
    atomic_add_t  atomic_add;
} gang_t;


// globals


static inline
void* malloc_zero(unsigned sz){
    void* p = malloc(sz);
    memset(p, 0, sz);
    return p;
}

static inline
void lock(gang_t* gang, volatile int *ptr){
    int old;
    while (1){
        old = gang->atomic_add(ptr, 1);
        if (old == 0) return;
        else gang->atomic_add(ptr, -1); // undo
    }
}

static inline
void unlock(gang_t* gang, volatile int *ptr){
    gang->atomic_add(ptr, -1);
}

static inline
gang_t* init_gang(int ncpu, kernel_t kernel, int ntid,
                  void *args, int arglen,
                  atomic_add_t atomic_add,
                  int *taskperqueue)
{
    *taskperqueue = ntid / ncpu + (ntid % ncpu ? 1 : 0);

    gang_t *gang  = malloc_zero(sizeof(gang_t));
    gang->len = ncpu;
    gang->done = 0;
    gang->members = malloc_zero(ncpu * sizeof(void*));
    gang->args = malloc_zero(arglen);
    gang->kernel = kernel;
    gang->ntid = ntid;
    gang->runct = 0;
    gang->atomic_add = atomic_add;

    memcpy(gang->args, args, arglen); // keep copy of the arguments

    return gang;
}

static inline
void init_workers(gang_t *gang, int sizeof_worker, int taskperqueue)
{
    int i;
    int ntask = 0;
    for (i = 0; i < gang->len; ++i) {
        worker_t *worker = malloc_zero(sizeof(pthread_worker_t));
        int begin = ntask;
        ntask += taskperqueue;
        int end = MIN(ntask, gang->ntid);

        worker->queue.begin = begin;
        worker->queue.end   = end;
        worker->queue.lock  = 0;
        worker->parent      = gang;

        gang->members[i] = (worker_t*)worker;
    }
}

static inline
gang_t* init_gang_workers(int ncpu, kernel_t kernel, int ntid,
                          void *args, int arglen, atomic_add_t atomic_add,
                          int sizeof_worker)
{
    int taskperqueue;
    gang_t* gang = init_gang(ncpu, kernel, ntid, args, arglen, atomic_add,
                             &taskperqueue);
    init_workers(gang, sizeof_worker, taskperqueue);
    return gang;
}

static inline
void fini_gang(gang_t *gang)
{
    if (gang->ntid != gang->runct) {
        printf("race condition detected: ntid=%d runct=%d\n",
               gang->ntid, gang->runct);
        exit(1);
    }
    free(gang->args);
    free(gang->members);
    free(gang);
}

static
void run_worker(worker_t *worker)
{
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
        worker->parent->kernel(begin, end, worker->parent->args);
        worker->parent->atomic_add(&worker->parent->runct, end - begin);
    }
    worker->parent->done += 1;

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
            worker->parent->kernel(begin, end, worker->parent->args);
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


void join_workers(gang_t *gang){ }


#else

#if defined(__WIN32__)
gang_t* start_workers(int ncpu, kernel_t kernel, int ntid, void *args,
                      int arglen, atomic_add_t atomic_add)
{
    int i;
    gang_t *gang = init_gang_workers(ncpu, kernel, ntid, args, arglen,
                                     atomic_add, sizeof(winthread_worker_t));

    for (i = 0; i < gang->len; ++i) {
        winthread_worker_t* worker = (winthread_worker_t*)gang->members[i];
        worker->thread = CreateThread(NULL, 0, (void*)run_worker,
                                      (void*)&worker->base, 0, NULL);
    }
    return gang;
}


void join_workers(gang_t *gang)
{
    int i;
    for (i = 0; i < gang->len; ++i){
        winthread_worker_t* worker = (winthread_worker_t*)gang->members[i];
        WaitForSingleObject(worker->thread, INFINITE);
        CloseHandle(worker->thread);
        free(worker);
    }
    fini_gang(gang);
}
#else
gang_t* start_workers(int ncpu, kernel_t kernel, int ntid, void *args,
                      int arglen, atomic_add_t atomic_add)
{
    int i;
    gang_t *gang = init_gang_workers(ncpu, kernel, ntid, args, arglen,
                                     atomic_add, sizeof(pthread_worker_t));

    for (i = 0; i < gang->len; ++i) {
        pthread_worker_t* worker = (pthread_worker_t*)gang->members[i];
        pthread_create(&worker->thread, NULL, (void*(*)(void*))run_worker,
                       &worker->base);
    }
    return gang;
}


void join_workers(gang_t *gang)
{
    int i;
    for (i = 0; i < gang->len; ++i){
        pthread_worker_t* worker = (pthread_worker_t*)gang->members[i];
        pthread_join(worker->thread, NULL);
        free(worker);
    }
    fini_gang(gang);
}
#endif

#endif // NOTHREAD

