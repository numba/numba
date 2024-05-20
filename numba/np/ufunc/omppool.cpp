/*
Threading layer on top of OpenMP.
*/

#include "../../_pymodule.h"
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#include <omp.h>
#include <string.h>
#include <stdio.h>
#include "workqueue.h"
#include "gufunc_scheduler.h"

#ifdef _WIN32
#include <malloc.h>
#else
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#endif

#define _DEBUG 0
#define _DEBUG_FORK 0

// OpenMP vendor strings
#if defined(_MSC_VER)
#define _OMP_VENDOR "MS"
#elif defined(__clang__)
#define _OMP_VENDOR "Intel"
#elif defined(__GNUC__) // NOTE: clang also defines this, but it's checked above
#ifndef _WIN32
#define _NOT_FORKSAFE 1 // GNU OpenMP Not forksafe
#endif
#define _OMP_VENDOR "GNU"
#endif

#if defined(_NOT_FORKSAFE)
static pid_t parent_pid = 0; // 0 is not set, users can't own this anyway
#endif


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
    return omp_get_thread_num();
}

static void
add_task(void *fn, void *args, void *dims, void *steps, void *data)
{
    puts("Running add_task() with omppool sequentially");
    typedef void (*func_ptr_t)(void *args, void *dims, void *steps, void *data);
    func_ptr_t func = reinterpret_cast<func_ptr_t>(fn);
    func(args, dims, steps, data);
}

static void
parallel_for(void *fn, char **args, size_t *dimensions, size_t *steps, void *data,
             size_t inner_ndim, size_t array_count, int num_threads)
{
    typedef void (*func_ptr_t)(char **args, size_t *dims, size_t *steps, void *data);
    func_ptr_t func = reinterpret_cast<func_ptr_t>(fn);
    static bool printed = false;
    if(!printed && _DEBUG)
    {
        puts("Using parallel_for");
        printed = true;
    }

#if defined(_NOT_FORKSAFE)
    // Handle GNU OpenMP not being forksafe...
    // This checks if the pid set by the process that initialized this library
    // matches the parent of this pid. If they do match this is a fork() from
    // Python and not a spawn(), as spawn()s reinit the library. Forks are
    // dangerous as GNU OpenMP is not forksafe, so warn then terminate.
    if(_DEBUG_FORK)
    {
        printf("Registered parent pid=%d, my pid=%d, my parent pid=%d\n", parent_pid, getpid(), getppid());
    }
    if (parent_pid == getppid())
    {
        fprintf(stderr, "%s", "Terminating: fork() called from a process "
                "already using GNU OpenMP, this is unsafe.\n");
        raise(SIGTERM);
        return;
    }
#endif

    //     args = <ir.Argument '.1' of type i8**>,
    //     dimensions = <ir.Argument '.2' of type i64*>
    //     steps = <ir.Argument '.3' of type i64*>
    //     data = <ir.Argument '.4' of type i8*>

    const size_t arg_len = (inner_ndim + 1);
    // index variable in OpenMP 'for' statement must have signed integral type for MSVC
    const ptrdiff_t size = (ptrdiff_t)dimensions[0];

    // holds the shared variable for `num_threads`, this is a bit superfluous
    // but present to force thinking about the scope of validity
    int agreed_nthreads = num_threads;

    if(_DEBUG)
    {
        printf("inner_ndim: %zu\n",inner_ndim);
        printf("arg_len: %zu\n", arg_len);
        printf("total: %td\n", size);
        printf("dimensions: ");
        for(size_t j = 0; j < arg_len; j++)
            printf("%zu, ", ((size_t *)dimensions)[j]);
        printf("\nsteps: ");
        for(size_t j = 0; j < array_count; j++)
            printf("%zu, ", steps[j]);
        printf("\n*args: ");
        for(size_t j = 0; j < array_count; j++)
            printf("%p, ", (void *)args[j]);
        printf("\n");
    }

    // Set the thread mask on the pragma such that the state is scope limited
    // and passed via a register on the OMP region call site, this limiting
    // global state and racing
    #pragma omp parallel num_threads(num_threads), shared(agreed_nthreads)
    {
        size_t * count_space = (size_t *)alloca(sizeof(size_t) * arg_len);
        char ** array_arg_space = (char**)alloca(sizeof(char*) * array_count);

        // tell the active thread team about the number of threads
        set_num_threads(agreed_nthreads);

        #pragma omp for
        for(ptrdiff_t r = 0; r < size; r++)
        {
            memcpy(count_space, dimensions, arg_len * sizeof(size_t));
            count_space[0] = 1;

            if(_DEBUG)
            {
                printf("THREAD %p:", count_space);
                printf("count_space: ");
                for(size_t j = 0; j < arg_len; j++)
                    printf("%zd, ", count_space[j]);
                printf("\n");
            }
            for(size_t j = 0; j < array_count; j++)
            {
                char * base = args[j];
                size_t step = steps[j];
                ptrdiff_t offset = step * r;
                array_arg_space[j] = base + offset;

                if(0&&_DEBUG)
                {
                    printf("Index %zu\n", j);
                    printf("-->Got base %p\n", (void *)base);
                    printf("-->Got step %zu\n", step);
                    printf("-->Got offset %td\n", offset);
                    printf("-->Got addr %p\n", (void *)array_arg_space[j]);
                }
            }

            if(_DEBUG)
            {
                printf("array_arg_space: ");
                for(size_t j = 0; j < array_count; j++)
                    printf("%p, ", (void *)array_arg_space[j]);
                printf("\n");
            }
            func(array_arg_space, count_space, steps, data);
        }
    }
}

static void launch_threads(int count)
{
    // this must be called in a fork+thread safe region from Python
    static bool initialized = false;
#ifdef _NOT_FORKSAFE
    parent_pid = getpid(); // record the parent PID for use later
    if(_DEBUG_FORK)
    {
        printf("Setting parent as %d\n", parent_pid);
    }
#endif
    if(initialized)
        return;
    if(_DEBUG)
        puts("Using OpenMP");
    if(count < 1)
        return;
    omp_set_num_threads(count);
    omp_set_nested(0x1); // enable nesting, control depth with OMP env var
    _INIT_NUM_THREADS = count;
}

static void synchronize(void)
{
}

static void ready(void)
{
}

MOD_INIT(omppool)
{
    PyObject *m;
    MOD_DEF(m, "omppool", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    SetAttrStringFromVoidPointer(m, launch_threads);
    SetAttrStringFromVoidPointer(m, synchronize);
    SetAttrStringFromVoidPointer(m, ready);
    SetAttrStringFromVoidPointer(m, add_task);
    SetAttrStringFromVoidPointer(m, parallel_for);
    SetAttrStringFromVoidPointer(m, do_scheduling_signed);
    SetAttrStringFromVoidPointer(m, do_scheduling_unsigned);
    SetAttrStringFromVoidPointer(m, set_num_threads);
    SetAttrStringFromVoidPointer(m, get_num_threads);
    SetAttrStringFromVoidPointer(m, get_thread_id);
    SetAttrStringFromVoidPointer(m, set_parallel_chunksize);
    SetAttrStringFromVoidPointer(m, get_parallel_chunksize);
    SetAttrStringFromVoidPointer(m, get_sched_size);
    SetAttrStringFromVoidPointer(m, allocate_sched);
    SetAttrStringFromVoidPointer(m, deallocate_sched);

    PyObject *tmp = PyString_FromString(_OMP_VENDOR);
    PyObject_SetAttrString(m, "openmp_vendor", tmp);
    Py_DECREF(tmp);

    return MOD_SUCCESS_VAL(m);
}
