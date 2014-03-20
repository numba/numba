/*
 * This is a sample module exposing numpy math functions needed by numba.
 *
 * The module unique content will be a property containing a vector of tuples.
 * Each tuple will hold (symbol_name, function_pointer).
 */


#include "_pymodule.h"


struct npy_math_entry {
    const char* name;
    void* func;
};

struct npy_math_entry exports[] = {
    { "test", (void*) 42 }
};


PyObject*
create_symbol_list()
{
    /*
     * note: reference stealing at its best
     * returns a PyList with a tuple for each symbol. The PyList has one reference.
     */
    size_t count = sizeof(exports) / sizeof(exports[0]);
    PyObject* pylist = PyList_New(count);

    for (size_t i = 0; i < count; ++i) {
        /* create the tuple */
        PyObject* tuple = Py_BuildValue("(s,l)", exports[i].name, exports[i].func);
        PyList_SET_ITEM(pylist, i, tuple);
    }

    return pylist;
}

MOD_INIT(_npymath_exports) {
    PyObject *module;
    MOD_DEF(module, "_npymath_exports", "No docs", NULL)
    if (!module) {
        return MOD_ERROR_VAL;
    }

    PyModule_AddObject(module, "symbols", create_symbol_list());

    return MOD_SUCCESS_VAL(module);
}


