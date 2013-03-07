#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


typedef void (*kernel_t)(void* args);

typedef struct queue {
    int begin, end;
} queue_t;

struct gang; // forward declaration

typedef struct worker {
    kernel_t        kernel;
    queue_t         queue;
    struct gang    *parent;
} worker_t;

typedef struct pthread_worker {
    worker_t    base;
    pthread_t   thread;
} pthread_worker_t;

typedef struct gang{
    worker_t   **members;
    int          len;
    void        *args;
    int          arglen;
} gang_t;


static
void run_worker(worker_t *worker)
{
    int arglen = worker->parent->arglen;
    int offset = 2;
    void *args = malloc(sizeof(int) * offset + arglen);
    int *arghead = (int*)args;
    memcpy(arghead + offset, worker->parent->args, arglen);
    arghead[0] = worker->queue.begin;
    arghead[1] = worker->queue.end;
    printf("in thread %d : %d\n", arghead[0], arghead[1]);
    worker->kernel(args);
    free(args);
}

gang_t* start_workers(int ncpu, kernel_t kernel, int ntid, void *args, int arglen)
{
    int i;
    int ntask = 0;
    int taskperqueue = ntid / ncpu + (ntid % ncpu ? 1 : 0);
    printf("ntid %d\n", ntid);
    printf("taskperqueue %d\n", taskperqueue);

    gang_t *gang  = malloc(sizeof(gang_t));
    gang->len = ncpu;
    gang->members = malloc(ncpu * sizeof(pthread_worker_t*));
    gang->args = malloc(arglen);
    gang->arglen = arglen;
    memcpy(gang->args, args, arglen); // keep copy of the arguments

    for (i = 0; i < gang->len; ++i) {
        pthread_worker_t *worker = malloc(sizeof(pthread_worker_t));
        int begin = ntask;
        ntask += taskperqueue;
        int end = MIN(ntask, ntid);
        worker->base.kernel      = kernel;
        worker->base.queue.begin = begin;
        worker->base.queue.end   = end;
        worker->base.parent      = gang;
        
        printf("i=%d begin=%d end=%d\n",
               i, worker->base.queue.begin, worker->base.queue.end);

        gang->members[i] = (worker_t*)worker;
    }

    for (i = 0; i < gang->len; ++i) {
        pthread_worker_t* worker = (pthread_worker_t*)gang->members[i];
        pthread_create(&worker->thread, NULL, (void*(*)(void*))run_worker,
                       &worker->base);
    }
    return gang;
}


void join_workers(gang_t *gang)
{
    printf("start join\n");
    int i;
    for (i = 0; i < gang->len; ++i){
    
        printf("joinning %d\n", i);
        pthread_worker_t* worker = (pthread_worker_t*)gang->members[i];
        pthread_join(worker->thread, NULL);
        free(worker);
    }
    free(gang->args);
    free(gang->members);
    free(gang);
}



//
//void launch(int ntid, kernel_t kernel, void *args)
//{
//    int tid;
//    int step = 4;
//    for (tid = 0; tid < ntid; tid += step){
//        ((int*)args)[0] = tid;             // set begin
//        ((int*)args)[1] = MIN(ntid, tid + step);      // set end
//        kernel(args);
//    }
//}
