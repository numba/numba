/*
Implement parallel vectorize workqueue on top of Intel TBB.
*/

#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#include <tbb/tbb.h>
#include <string.h>
#include <stdio.h>
#include "workqueue.h"
#include "../_pymodule.h"

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


void ignore_blocking_terminate_assertion( const char*, int, const char*, const char * ) {
    tbb::internal::runtime_warning("Unable to wait for threads to shut down before fork(). It can break multithreading in child process\n");
}
void ignore_assertion( const char*, int, const char*, const char * ) {}

static void prepare_fork(void) {
    //puts("Suspending TBB: prepare fork");
    if(tsi) {
        assertion_handler_type orig = tbb::set_assertion_handler(ignore_blocking_terminate_assertion);
        tsi->terminate();
        tbb::set_assertion_handler(orig);
    }
}

static void reset_after_fork(void) {
    //puts("Resuming TBB: after fork");
    if(tsi)
        tsi->initialize(tsi_count);
}

static void unload_tbb(void) {
    if(tsi) {
        delete tg;
        tg = NULL;
        //puts("Unloading TBB");
        assertion_handler_type orig = tbb::set_assertion_handler(ignore_assertion);
        tsi->terminate();
        tbb::set_assertion_handler(orig);
        delete tsi;
        tsi = NULL;
    }
}

static void launch_threads(int count) {
    if(tsi)
        return;
    if(count < 1)
        count = tbb::task_scheduler_init::automatic;
#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
    tsi = new tbb::task_scheduler_init(tsi_count = count, 0, /*blocking termination*/true);
#else
#error This version of TBB does not support blocking terminate or implements it differently
#endif
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

    return MOD_SUCCESS_VAL(m);
}
