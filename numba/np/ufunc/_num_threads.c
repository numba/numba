// Thread local num_threads variable for masking out the total number of
// launched threads.

#include "../../_pymodule.h"

#ifdef _MSC_VER
#define THREAD_LOCAL(ty) __declspec(thread) ty
#else
/* Non-standard C99 extension that's understood by gcc and clang */
#define THREAD_LOCAL(ty) __thread ty
#endif

static THREAD_LOCAL(int) num_threads = 0;

static void set_num_threads(int count)
{
    num_threads = count;
}

static int get_num_threads(void)
{
    return num_threads;
}

MOD_INIT(_num_threads)
{
    PyObject *m;
    MOD_DEF(m, "_num_threads", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    SetAttrStringFromVoidPointer(m, set_num_threads);
    SetAttrStringFromVoidPointer(m, get_num_threads);

    return MOD_SUCCESS_VAL(m);
}
