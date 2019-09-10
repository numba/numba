#include "_numba_common.h"
/* Define all runtime-required symbols in this C module, but do not
   export them outside the shared library if possible. */

#define NUMBA_EXPORT_FUNC(_rettype) VISIBILITY_HIDDEN _rettype
#define NUMBA_EXPORT_DATA(_vartype) VISIBILITY_HIDDEN _vartype



// #define NUMBA_EXPORT_FUNC(_rettype) static _rettype
// #define NUMBA_EXPORT_DATA(_vartype) static _vartype


// #define NUMBA_EXPORT_FUNC(_rettype) _rettype
// #define NUMBA_EXPORT_DATA(_vartype) _vartype
