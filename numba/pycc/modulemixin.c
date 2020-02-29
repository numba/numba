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
#include "../_dynfunc.c"

#if PYCC_USE_NRT
#include "../core/runtime/_nrt_python.c"
#include "../core/runtime/nrt.h"
#endif


/* NOTE: import_array() is macro, not a function.  It returns NULL on
   failure */
static void *
wrap_import_array(void) {
    import_array();
    return (void *) 1;
}


static int
init_numpy(void) {
    return wrap_import_array() != NULL;
}


#ifndef PYCC_MODULE_NAME
#error PYCC_MODULE_NAME must be defined
#endif

/* Preprocessor trick: need to use two levels of macros otherwise
   PYCC_MODULE_NAME would not get expanded */
#define __PYCC(prefix, modname) prefix ## modname
#define _PYCC(prefix, modname) __PYCC(prefix, modname)
#define PYCC(prefix) _PYCC(prefix, PYCC_MODULE_NAME)

/* Silence warnings about unused functions */
VISIBILITY_HIDDEN void **PYCC(_unused_) = {
    (void *) Numba_make_generator,
};

/* The LLVM-generated functions for atomic refcounting */
extern void *nrt_atomic_add, *nrt_atomic_sub;

/* The structure type constructed by PythonAPI.serialize_uncached() */
typedef struct {
    const char *data;
    int len;
} env_def_t;

/* Environment GlobalVariable address type */
typedef void **env_gv_t;

/*
 * Recreate an environment object from a env_def_t structure.
 */
static EnvironmentObject *
recreate_environment(PyObject *module, env_def_t env)
{
    EnvironmentObject *envobj;
    PyObject *env_consts;

    env_consts = numba_unpickle(env.data, env.len);
    if (env_consts == NULL)
        return NULL;
    if (!PyList_Check(env_consts)) {
        PyErr_Format(PyExc_TypeError,
                     "environment constants should be a list, got '%s'",
                     Py_TYPE(env_consts)->tp_name);
        Py_DECREF(env_consts);
        return NULL;
    }

    envobj = env_new_empty(&EnvironmentType);
    if (envobj == NULL) {
        Py_DECREF(env_consts);
        return NULL;
    }
    envobj->consts = env_consts;
    envobj->globals = PyModule_GetDict(module);
    if (envobj->globals == NULL) {
        Py_DECREF(envobj);
        return NULL;
    }
    Py_INCREF(envobj->globals);
    return envobj;
}

/*
 * Subroutine to initialize all resources required for running the
 * pycc-compiled functions.
 */

int
PYCC(pycc_init_) (PyObject *module, PyMethodDef *defs,
                                    env_def_t *envs,
                                    env_gv_t *envgvs)
{
    PyMethodDef *fdef;
    PyObject *modname = NULL;
    PyObject *docobj = NULL;
    int i;

    if (!init_numpy()) {
        goto error;
    }
    if (init_dynfunc_module(module)) {
        goto error;
    }
    /* Initialize random generation. */
    numba_rnd_ensure_global_init();

#if PYCC_USE_NRT
    NRT_MemSys_init();
    NRT_MemSys_set_atomic_inc_dec((NRT_atomic_inc_dec_func) &nrt_atomic_add,
                                  (NRT_atomic_inc_dec_func) &nrt_atomic_sub);
    if (init_nrt_python_module(module)) {
        goto error;
    }
#endif

    modname = PyObject_GetAttrString(module, "__name__");
    if (modname == NULL) {
        goto error;
    }

    /* Empty docstring for all compiled functions */
    docobj = PyString_FromString("");
    if (docobj == NULL) {
        goto error;
    }

    /* Overwrite C method objects with our own Closure objects, in order
     * to make their environments available to the compiled functions.
     */
    for (i = 0, fdef = defs; fdef->ml_name != NULL; i++, fdef++) {
        PyObject *func;
        PyObject *nameobj;
        EnvironmentObject *envobj;

        envobj = recreate_environment(module, envs[i]);
        if (envobj == NULL) {
            goto error;
        }
        nameobj = PyString_FromString(fdef->ml_name);
        if (nameobj == NULL) {
            Py_DECREF(envobj);
            goto error;
        }
        // Store the environment pointer into the global
        *envgvs[i] = envobj;

        func = pycfunction_new(module, nameobj, docobj,
                               fdef->ml_meth, envobj, NULL);
        Py_DECREF(envobj);
        Py_DECREF(nameobj);

        if (func == NULL) {
            goto error;
        }
        if (PyObject_SetAttrString(module, fdef->ml_name, func)) {
            Py_DECREF(func);
            goto error;
        }
        Py_DECREF(func);
    }
    Py_DECREF(docobj);
    Py_DECREF(modname);
    return 0;

error:
    Py_XDECREF(docobj);
    Py_XDECREF(modname);
    return -1;
}
