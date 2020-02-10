typedef struct opaque_thread * thread_pointer;

enum QUEUE_STATE
{
    /*
    The queue has 4 states:

    IDLE: not doing anything
    READY: tasks enqueued; signal workers to start
    RUNNING: workers running
    DONE: workers completed
    */
    IDLE = 0, READY, RUNNING, DONE
};

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

/* parallel for loop with 1d tiling.

 Args:

 fn - function pointer to the gufunc

 These are from the Python: args, dimensions, steps, data = lfunc.args
 args - gufunc args
 dims - gufunc dims
 steps - gufunc steps
 data - gufunc data

 inner_ndim - inner dimension of the gufunc
 array_count - the number of arrays in the signature (Python: len(sig.args) + 1)
 the +1 is for the output array.

 */
static void
parallel_for(void *fn, char **args, size_t *dims, size_t *steps, void *data,\
             size_t inner_ndim, size_t array_count);
