/*
Threading layer on top of OpenMP.
*/

#include <omp.h>
#include <string.h>
#include <stdio.h>
#include "workqueue.h"
#include "../_pymodule.h"
#include "gufunc_scheduler.h"

#ifdef _MSC_VER
#include <malloc.h>
#endif

static void
add_task(void *fn, void *args, void *dims, void *steps, void *data) {
    puts("Running add_task() with omppool sequentially");
    auto func = reinterpret_cast<void (*)(void *args, void *dims, void *steps, void *data)>(fn);
    func(args, dims, steps, data);
}

#define _DEBUG 0

static void
parallel_for(void *fn, char **args, size_t *dimensions, size_t *steps, void *data,
                size_t inner_ndim, size_t array_count, size_t)
{
    auto func = reinterpret_cast<void (*)(char **args, size_t *dims, size_t *steps, void *data)>(fn);
    static bool printed = false;
    if(!printed) {
        puts("Using parallel_for");
        printed = true;
    }

    //     args = <ir.Argument '.1' of type i8**>,
    //     dimensions = <ir.Argument '.2' of type i64*>
    //     steps = <ir.Argument '.3' of type i64*>
    //     data = <ir.Argument '.4' of type i8*>

    const size_t arg_len = (inner_ndim + 1);
    // index variable in OpenMP 'for' statement must have signed integral type for MSVC
    const ptrdiff_t size = (ptrdiff_t)dimensions[0];

    if(_DEBUG)
    {
        printf("inner_ndim: %lu\n",inner_ndim);
        printf("arg_len: %lu\n", arg_len);
        printf("total: %ld\n", size);
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

    #pragma omp parallel for
    for(ptrdiff_t r = 0; r < size; r++) {
        size_t * count_space = (size_t *)alloca(sizeof(size_t) * arg_len);
        char ** array_arg_space = (char**)alloca(sizeof(char*) * array_count);
        memcpy(count_space, dimensions, arg_len * sizeof(size_t));
        count_space[0] = 1;

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
            ptrdiff_t offset = step * r;
            array_arg_space[j] = base + offset;

            if(0&&_DEBUG)
            {
                printf("Index %lu\n", j);
                printf("-->Got base %p\n", (void *)base);
                printf("-->Got step %lu\n", step);
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
        func(array_arg_space, count_space, steps, data);
    }
}

static void launch_threads(int count) {
    static bool initialized = false;
    if(initialized)
        return;
    puts("Using OpenMP");
    if(count < 1)
        return;
    omp_set_num_threads(count);
}

static void synchronize(void) {
}

static void ready(void) {
}

MOD_INIT(omppool) {
    PyObject *m;
    MOD_DEF(m, "omppool", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

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


    return MOD_SUCCESS_VAL(m);
}
