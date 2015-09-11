typedef struct opaque_thread * thread_pointer;

enum QUEUE_STATE {
    /*
    The queue has 4 states:

    IDLE: not doing anything
    READY: tasks enqueued; signal workers to start
    RUNNING: workers running
    DONE: workers completed
    */
    IDLE = 0, READY, RUNNING, DONE
};

/*
sleep for a short time
*/
static
void take_a_nap(int usec);


/*
CAS: compare and swap function

It is generated from parallel.py by LLVM to get a portable CAS function.
*/
typedef int cas_function_t(volatile int *ptr, int old, int val);

/*
Do CAS until successful.
Spin until the value in `ptr` changes from `old` to `repl`.
Takes a 1us nap in after each CAS failure.
*/
static
void cas_wait(volatile int *ptr, const int old, const int repl);

/* Launch new thread */
static
thread_pointer numba_new_thread(void *worker, void *arg);

/* Set CAS function.
Note: During interpreter teardown, LLVM will release all function memory.
      To protect against the fault due to calling non-executable memory,
      Call set_cas(NULL) to disable the workqueue.
*/
static
void set_cas(void *ptr);

/* Launch `count` number of threads and create the associated thread queue.
Must invoke once before each add_task() is used.
*Warning* queues memory are leaked at interpreter tear down!
*/
static
void launch_threads(int count);

/* Add task to queue
Automatically assigned to queues of different thread in a round robin fashion.
*/
static
void add_task(void *fn, void *args, void *dims, void *steps, void *data);

/* Wait until all tasks are done */
static
void synchronize(void);

/* Signal worker threads that tasks are added and it is ready to run */
static
void ready(void);
