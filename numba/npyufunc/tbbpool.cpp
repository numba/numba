/*
Implement parallel vectorize workqueue on top of Intel TBB.
*/

#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#include <tbb/tbb.h>
#include <string.h>
#include <stdio.h>
#include "workqueue.h"
#include "../_pymodule.h"
#include "gufunc_scheduler.h"

#if TBB_INTERFACE_VERSION >= 9106
    #define TSI_INIT(count) tbb::task_scheduler_init(count)
    #define TSI_TERMINATE(tsi) tsi->blocking_terminate(std::nothrow)
#else
#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
    #define TSI_INIT(count) tbb::task_scheduler_init(count, 0, /*blocking termination*/true)
    #define TSI_TERMINATE(tsi) tsi->terminate()
#else
#error This version of TBB does not support blocking terminate
#endif
#endif

static tbb::task_group *tg = NULL;
static tbb::task_scheduler_init *tsi = NULL;
static int tsi_count = 0;

static void
add_task(void *fn, void *args, void *dims, void *steps, void *data) {
    tg->run([=]{
        auto func = reinterpret_cast<void (*)(void *args, void *dims, void *steps, void *data)>(fn);
        func(args, dims, steps, data);
    });
}

#define _DEBUG 0

static void
parallel_for_1d(void *fn, char **args, size_t *dimensions, size_t *steps, void *data,
                size_t inner_ndim, size_t array_count, size_t)
{
    static bool printed = false;
    if(!printed) {
        puts("Using parallel_for_1d");
        printed = true;
    }

    //     args = <ir.Argument '.1' of type i8**>,
    //     dimensions = <ir.Argument '.2' of type i64*>
    //     steps = <ir.Argument '.3' of type i64*>
    //     data = <ir.Argument '.4' of type i8*>

    const size_t arg_len = (inner_ndim + 1);

    if(_DEBUG)
    {
        printf("inner_ndim: %ld\n",inner_ndim);
        printf("arg_len: %ld\n", arg_len);
        printf("total: %ld\n", dimensions[0]);
        printf("dimensions: ");
        for(size_t j = 0; j < arg_len; j++)
            printf("%ld, ", ((size_t *)dimensions)[j]);
        printf("\nsteps: ");
        for(size_t j = 0; j < array_count; j++)
            printf("%ld, ", steps[j]);
        printf("\n*args: ");
        for(size_t j = 0; j < array_count; j++)
            printf("%p, ", (void *)args[j]);
        printf("\n");
    }

    using range_t = tbb::blocked_range<size_t>;
    tbb::parallel_for(range_t(0, dimensions[0]), [=](const range_t &range) {
        size_t * count_space = (size_t *)alloca(sizeof(size_t) * arg_len);
        char ** array_arg_space = (char**)alloca(sizeof(char*) * array_count);
        memcpy(count_space, dimensions, arg_len * sizeof(size_t));
        count_space[0] = range.size();

        if(_DEBUG)
        {
            printf("THREAD %p:", count_space);
            printf("count_space: ");
            for(size_t j = 0; j < arg_len; j++)
                printf("%ld, ", count_space[j]);
            printf("\n");
        }
        for(size_t j = 0; j < array_count; j++)
        {
            char * base = args[j];
            size_t step = steps[j];
            ptrdiff_t offset = step * range.begin();
            array_arg_space[j] = (char *)(base + offset);

            if(0&&_DEBUG)
            {
                printf("Index %ld\n", j);
                printf("-->Got base %p\n", (void *)base);
                printf("-->Got step %ld\n", step);
                printf("-->Got offset %ld\n", offset);
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
        auto func = reinterpret_cast<void (*)(void *args, void *dims, void *steps, void *data)>(fn);
        func((void *)array_arg_space, (void *)count_space, steps, data);
    });
}

void ignore_blocking_terminate_assertion( const char*, int, const char*, const char * ) {
    tbb::internal::runtime_warning("Unable to wait for threads to shut down before fork(). It can break multithreading in child process\n");
}
void ignore_assertion( const char*, int, const char*, const char * ) {}

static void prepare_fork(void) {
    puts("Suspending TBB: prepare fork");
    if(tsi) {
        assertion_handler_type orig = tbb::set_assertion_handler(ignore_blocking_terminate_assertion);
        TSI_TERMINATE(tsi);
        tbb::set_assertion_handler(orig);
    }
}

static void reset_after_fork(void) {
    puts("Resuming TBB: after fork");
    if(tsi)
        tsi->initialize(tsi_count);
}

static void unload_tbb(void) {
    if(tsi) {
        delete tg;
        tg = NULL;
        puts("Unloading TBB");
        assertion_handler_type orig = tbb::set_assertion_handler(ignore_assertion);
        TSI_TERMINATE(tsi);
        tbb::set_assertion_handler(orig);
        delete tsi;
        tsi = NULL;
    }
}

static void launch_threads(int count) {
    puts("Using TBB");
    if(tsi)
        return;
    if(count < 1)
        count = tbb::task_scheduler_init::automatic;
    tsi = new TSI_INIT(tsi_count = count);
    tg = new tbb::task_group;
    tg->run([]{}); // start creating threads asynchronously

#ifndef _MSC_VER
    pthread_atfork(prepare_fork, reset_after_fork, reset_after_fork);
#endif
}

static void synchronize(void) {
    tg->wait();
}

static void ready(void) {
}

MOD_INIT(workqueue) {
    PyObject *m;
    MOD_DEF(m, "workqueue", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;
#if PY_MAJOR_VERSION >= 3
    PyModuleDef *md = PyModule_GetDef(m);
    if (md) {
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
    PyObject_SetAttrString(m, "parallel_for_1d",
                           PyLong_FromVoidPtr((void*)&parallel_for_1d));
    PyObject_SetAttrString(m, "do_scheduling_signed",
                           PyLong_FromVoidPtr((void*)&do_scheduling_signed));
    PyObject_SetAttrString(m, "do_scheduling_unsigned",
                           PyLong_FromVoidPtr((void*)&do_scheduling_unsigned));


    return MOD_SUCCESS_VAL(m);
}
