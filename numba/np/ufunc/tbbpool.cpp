/*
Implement parallel vectorize workqueue on top of Intel TBB.
*/

#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
/* tbb.h redefines these */
#include "../../_pymodule.h"
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <tbb/tbb.h>
#include <string.h>
#include <stdio.h>
#include "workqueue.h"

#include "gufunc_scheduler.h"

/* TBB 2019 U5 is the minimum required version as this is needed:
 * https://github.com/intel/tbb/blob/18070344d755ece04d169e6cc40775cae9288cee/CHANGES#L133-L134
 * and therefore
 * https://github.com/intel/tbb/blob/18070344d755ece04d169e6cc40775cae9288cee/CHANGES#L128-L129
 * from here:
 * https://github.com/intel/tbb/blob/2019_U5/include/tbb/tbb_stddef.h#L29
 */
#if TBB_INTERFACE_VERSION < 11006
#error "TBB version is too old, 2019 update 5, i.e. TBB_INTERFACE_VERSION >= 11005 required"
#endif

#define TSI_INIT(count) tbb::task_scheduler_init(count)
#define TSI_TERMINATE(tsi) tsi->blocking_terminate(std::nothrow)

#define _DEBUG 0
#define _TRACE_SPLIT 0

static tbb::task_group *tg = NULL;
static tbb::task_scheduler_init *tsi = NULL;
static int tsi_count = 0;

#ifdef _MSC_VER
#define THREAD_LOCAL(ty) __declspec(thread) ty
#else
/* Non-standard C99 extension that's understood by gcc and clang */
#define THREAD_LOCAL(ty) __thread ty
#endif

// This is the number of threads that is default, it is set on initialisation of
// the threading backend via the launch_threads() call
static int _INIT_NUM_THREADS = -1;

// This is the per-thread thread mask, each thread can carry its own mask.
static THREAD_LOCAL(int) _TLS_num_threads = 0;


static void
set_num_threads(int count)
{
    _TLS_num_threads = count;
}

static int
get_num_threads(void)
{
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
    return tbb::task_arena::current_thread_index();
}

// watch the arena, if it decides to create more threads/add threads into the
// arena then make sure they get the right thread count
class fix_tls_observer: public tbb::task_scheduler_observer {
    int mask_val;
    void on_scheduler_entry( bool is_worker ) override;
public:
    fix_tls_observer(tbb::task_arena &arena, int mask) : tbb::task_scheduler_observer(arena), mask_val(mask)
    {
        observe(true);
    }
};

void fix_tls_observer::on_scheduler_entry(bool worker) {
    set_num_threads(mask_val);
}

static void
add_task(void *fn, void *args, void *dims, void *steps, void *data)
{
    tg->run([=]
    {
        auto func = reinterpret_cast<void (*)(void *args, void *dims, void *steps, void *data)>(fn);
        func(args, dims, steps, data);
    });
}

static void
parallel_for(void *fn, char **args, size_t *dimensions, size_t *steps, void *data,
             size_t inner_ndim, size_t array_count, int num_threads)
{
    static bool printed = false;
    if(!printed && _DEBUG)
    {
        puts("Using parallel_for");
        printed = true;
    }

    //     args = <ir.Argument '.1' of type i8**>,
    //     dimensions = <ir.Argument '.2' of type i64*>
    //     steps = <ir.Argument '.3' of type i64*>
    //     data = <ir.Argument '.4' of type i8*>

    const size_t arg_len = (inner_ndim + 1);

    if(_DEBUG && _TRACE_SPLIT)
    {
        printf("inner_ndim: %lu\n",inner_ndim);
        printf("arg_len: %lu\n", arg_len);
        printf("total: %lu\n", dimensions[0]);
        printf("dimensions: ");
        for(size_t j = 0; j < arg_len; j++)
            printf("%lu, ", ((size_t *)dimensions)[j]);
        printf("\nsteps: ");
        for(size_t j = 0; j < array_count; j++)
            printf("%lu, ", steps[j]);
        printf("\n*args: ");
        for(size_t j = 0; j < array_count; j++)
            printf("%p, ", (void *)args[j]);
        printf("\n");
    }

    // This is making the assumption that the calling thread knows the truth
    // about num_threads, which should be correct via the following:
    // program starts/reinits and the threadpool launches, num_threads TLS is
    // set as default. Any thread spawned on init making a call to this function
    // will have a valid num_threads TLS slot and so the task_arena is sized
    // appropriately and it's value is used in the observer that fixes the TLS
    // slots of any subsequent threads joining the task_arena. This leads to
    // all threads in a task_arena having valid num_threads TLS slots prior to
    // doing any work. Any further call to query the TLS slot value made by any
    // thread in the arena is then safe and were any thread to create a nested
    // parallel region the same logic applies as per program start/reinit.
    tbb::task_arena limited(num_threads);
    fix_tls_observer observer(limited, num_threads);

    limited.execute([&]{
        using range_t = tbb::blocked_range<size_t>;
        tbb::parallel_for(range_t(0, dimensions[0]), [=](const range_t &range)
        {
            size_t * count_space = (size_t *)alloca(sizeof(size_t) * arg_len);
            char ** array_arg_space = (char**)alloca(sizeof(char*) * array_count);
            memcpy(count_space, dimensions, arg_len * sizeof(size_t));
            count_space[0] = range.size();

            if(_DEBUG && _TRACE_SPLIT > 1)
            {
                printf("THREAD %p:", count_space);
                printf("count_space: ");
                for(size_t j = 0; j < arg_len; j++)
                    printf("%lu, ", count_space[j]);
                printf("\n");
            }
            for(size_t j = 0; j < array_count; j++)
            {
                char * base = args[j];
                size_t step = steps[j];
                ptrdiff_t offset = step * range.begin();
                array_arg_space[j] = base + offset;

                if(_DEBUG && _TRACE_SPLIT > 2)
                {
                    printf("Index %ld\n", j);
                    printf("-->Got base %p\n", (void *)base);
                    printf("-->Got step %lu\n", step);
                    printf("-->Got offset %ld\n", offset);
                    printf("-->Got addr %p\n", (void *)array_arg_space[j]);
                }
            }

            if(_DEBUG && _TRACE_SPLIT > 2)
            {
                printf("array_arg_space: ");
                for(size_t j = 0; j < array_count; j++)
                    printf("%p, ", (void *)array_arg_space[j]);
                printf("\n");
            }
            auto func = reinterpret_cast<void (*)(char **args, size_t *dims, size_t *steps, void *data)>(fn);
            func(array_arg_space, count_space, steps, data);
        });
    });
}

void ignore_blocking_terminate_assertion( const char*, int, const char*, const char * )
{
    tbb::internal::runtime_warning("Unable to wait for threads to shut down before fork(). It can break multithreading in child process\n");
}

void ignore_assertion( const char*, int, const char*, const char * ) {}

static void prepare_fork(void)
{
    if(_DEBUG)
    {
        puts("Suspending TBB: prepare fork");
    }
    if(tsi)
    {
        assertion_handler_type orig = tbb::set_assertion_handler(ignore_blocking_terminate_assertion);
        TSI_TERMINATE(tsi);
        tbb::set_assertion_handler(orig);
    }
}

static void reset_after_fork(void)
{
    if(_DEBUG)
    {
        puts("Resuming TBB: after fork");
    }
    if(tsi)
        tsi->initialize(tsi_count);
}

#if PY_MAJOR_VERSION >= 3
static void unload_tbb(void)
{
    if(tsi)
    {
        if(_DEBUG)
        {
            puts("Unloading TBB");
        }
        tg->wait();
        delete tg;
        tg = NULL;
        assertion_handler_type orig = tbb::set_assertion_handler(ignore_assertion);
        tsi->terminate(); // no blocking terminate is needed here
        tbb::set_assertion_handler(orig);
        delete tsi;
        tsi = NULL;
    }
}
#endif

static void launch_threads(int count)
{
    if(tsi)
        return;
    if(_DEBUG)
        puts("Using TBB");
    if(count < 1)
        count = tbb::task_scheduler_init::automatic;
    tsi = new TSI_INIT(tsi_count = count);
    tg = new tbb::task_group;
    tg->run([] {}); // start creating threads asynchronously

    _INIT_NUM_THREADS = count;

#ifndef _MSC_VER
    pthread_atfork(prepare_fork, reset_after_fork, reset_after_fork);
#endif
}

static void synchronize(void)
{
    tg->wait();
}

static void ready(void)
{
}


MOD_INIT(tbbpool)
{
    PyObject *m;
    MOD_DEF(m, "tbbpool", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;
#if PY_MAJOR_VERSION >= 3
    PyModuleDef *md = PyModule_GetDef(m);
    if (md)
    {
        md->m_free = (freefunc)unload_tbb;
    }
#endif

    PyObject_SetAttrString(m, "launch_threads",
                           PyLong_FromVoidPtr((void*)&launch_threads));
    PyObject_SetAttrString(m, "synchronize",
                           PyLong_FromVoidPtr((void*)&synchronize));
    PyObject_SetAttrString(m, "ready",
                           PyLong_FromVoidPtr((void*)&ready));
    PyObject_SetAttrString(m, "add_task",
                           PyLong_FromVoidPtr((void*)&add_task));
    PyObject_SetAttrString(m, "parallel_for",
                           PyLong_FromVoidPtr((void*)&parallel_for));
    PyObject_SetAttrString(m, "do_scheduling_signed",
                           PyLong_FromVoidPtr((void*)&do_scheduling_signed));
    PyObject_SetAttrString(m, "do_scheduling_unsigned",
                           PyLong_FromVoidPtr((void*)&do_scheduling_unsigned));
    PyObject_SetAttrString(m, "set_num_threads",
                           PyLong_FromVoidPtr((void*)&set_num_threads));
    PyObject_SetAttrString(m, "get_num_threads",
                           PyLong_FromVoidPtr((void*)&get_num_threads));
    PyObject_SetAttrString(m, "get_thread_id",
                           PyLong_FromVoidPtr((void*)&get_thread_id));

    return MOD_SUCCESS_VAL(m);
}
