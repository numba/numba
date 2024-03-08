/*
 * Copyright (c) 2017 Intel Corporation
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GUFUNC_SCHEDULER
#define GUFUNC_SCHEDULER

/* define int64_t and uint64_t for Visual Studio, where stdint only available > VS2008 */
#ifdef _MSC_VER
    #define int64_t signed __int64
    #define uint64_t unsigned __int64
#else
    #include <stdint.h>
#endif

#ifndef __SIZEOF_POINTER__
    /* MSVC doesn't define __SIZEOF_POINTER__ */
    #if defined(_WIN64)
        #define intp int64_t
        #define uintp uint64_t
    #elif defined(_WIN32)
        #define intp int
        #define uintp unsigned
    #else
        #error "cannot determine size of intp"
    #endif
#elif __SIZEOF_POINTER__ == 8
    #define intp int64_t
    #define uintp uint64_t
#else
    #define intp int
    #define uintp unsigned
#endif

#ifdef __cplusplus
extern "C"
{
#endif

void do_scheduling_signed(uintp num_dim, intp *starts, intp *ends, uintp num_threads, intp *sched, intp debug);
void do_scheduling_unsigned(uintp num_dim, intp *starts, intp *ends, uintp num_threads, uintp *sched, intp debug);
uintp set_parallel_chunksize(uintp);
uintp get_parallel_chunksize(void);
uintp get_sched_size(uintp num_threads, uintp num_dim, intp *starts, intp *ends);
intp * allocate_sched(uintp sched_size);
void deallocate_sched(intp * sched);

#ifdef __cplusplus
}
#endif

#endif
