/*
 * This C file is compiled and linked into pycc-generated shared objects.
 * It provides the Numba helper functions for runtime use in pycc-compiled
 * functions.
 */

#include "../_numba_common.h"
#include "../_pymodule.h"

/* Define all runtime-required symbols in this C module, but do not
   export them outside the shared library if possible. */

#define NUMBA_EXPORT_FUNC(_rettype) VISIBILITY_HIDDEN _rettype
#define NUMBA_EXPORT_DATA(_vartype) VISIBILITY_HIDDEN _vartype

#include "../_helperlib.c"
#include "../runtime/nrt.h"

/* NOTE: import_array() is macro, not a function.  It returns NULL on
   failure on py3, but nothing on py2. */
#if PY_MAJOR_VERSION >= 3
    static void *
    wrap_import_array(void) {
        import_array();
        return (void *) 1;
    }
#else
    static void
    wrap_import_array(void) {
        import_array();
    }
#endif


static int
init_numpy(void) {
    #if PY_MAJOR_VERSION >= 3
        return wrap_import_array() != NULL;
    #else
        wrap_import_array();
        return 1;   /* always succeed */
    #endif
}

#ifndef PYCC_MODULE_NAME
#error PYCC_MODULE_NAME must be defined
#endif

/* Preprocessor trick: need to use two levels of macros otherwise
   PYCC_MODULE_NAME would not get expanded */
#define _PYCC_INIT(name) pycc_init_##name
#define PYCC_INIT(name) _PYCC_INIT(name)

extern void *nrt_atomic_add, *nrt_atomic_sub;

void
PYCC_INIT(PYCC_MODULE_NAME) (PyObject *module)
{
    if (!init_numpy())
        Py_FatalError("Failed initializing numpy C API");
#if PYCC_USE_NRT
    NRT_MemSys_init();
    NRT_MemSys_set_atomic_inc_dec((NRT_atomic_inc_dec_func) &nrt_atomic_add,
                                  (NRT_atomic_inc_dec_func) &nrt_atomic_sub);
#endif
}
