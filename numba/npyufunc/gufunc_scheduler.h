#ifndef GUFUNC_SCHEDULER
#define GUFUNC_SCHEDULER

#include <stdint.h>

#ifndef __SIZEOF_POINTER__
    /* MSVC doesn't define __SIZEOF_POINTER__ */
    #if   defined(_WIN64)
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

void do_scheduling(intp num_dim, intp *dims, uintp num_threads, intp *sched, intp debug);

#ifdef __cplusplus
}
#endif

#endif
