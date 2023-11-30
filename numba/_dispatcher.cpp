#include "_pymodule.h"

#include <cstring>
#include <ctime>
#include <cassert>
#include <vector>

#include "_typeof.h"
#include "frameobject.h"
#include "traceback.h"
#include "core/typeconv/typeconv.hpp"
#include "_devicearray.h"

/*
 * Notes on the C_TRACE macro:
 *
 * The original C_TRACE macro (from ceval.c) would call
 * PyTrace_C_CALL et al., for which the frame argument wouldn't
 * be usable. Since we explicitly synthesize a frame using the
 * original Python code object, we call PyTrace_CALL instead so
 * the profiler can report the correct source location.
 *
 * Likewise, while ceval.c would call PyTrace_C_EXCEPTION in case
 * of error, the profiler would simply expect a RETURN in case of
 * a Python function, so we generate that here (making sure the
 * exception state is preserved correctly).
 *
 */

#if (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION == 12)
/* empty */
#elif (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION == 11)
#ifndef Py_BUILD_CORE
    #define Py_BUILD_CORE 1
#endif
#include "internal/pycore_frame.h"
#include "internal/pycore_pyerrors.h"

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/deaf509e8fc6e0363bd6f26d52ad42f976ec42f2/Python/ceval.c#L6804
 */
static int
call_trace(Py_tracefunc func, PyObject *obj,
           PyThreadState *tstate, PyFrameObject *frame,
           int what, PyObject *arg)
{
    int result;
    if (tstate->tracing) {
        return 0;
    }
    if (frame == NULL) {
        return -1;
    }
    int old_what = tstate->tracing_what;
    tstate->tracing_what = what;
    PyThreadState_EnterTracing(tstate);
    result = func(obj, frame, what, NULL);
    PyThreadState_LeaveTracing(tstate);
    tstate->tracing_what = old_what;
    return result;
}

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/d5650a1738fe34f6e1db4af5f4c4edb7cae90a36/Python/ceval.c#L4220-L4240
 */
static int
call_trace_protected(Py_tracefunc func, PyObject *obj,
                     PyThreadState *tstate, PyFrameObject *frame,
                     int what, PyObject *arg)
{
    PyObject *type, *value, *traceback;
    int err;
    _PyErr_Fetch(tstate, &type, &value, &traceback);
    err = call_trace(func, obj, tstate, frame, what, arg);
    if (err == 0)
    {
        _PyErr_Restore(tstate, type, value, traceback);
        return 0;
    }
    else {
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        return -1;
    }
}

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/deaf509e8fc6e0363bd6f26d52ad42f976ec42f2/Python/ceval.c#L7245
 * NOTE: The state test https://github.com/python/cpython/blob/d5650a1738fe34f6e1db4af5f4c4edb7cae90a36/Python/ceval.c#L4521
 * has been removed, it's dealt with in call_cfunc.
 */
#define C_TRACE(x, call, frame) \
if (call_trace(tstate->c_profilefunc, tstate->c_profileobj, \
    tstate, frame, \
    PyTrace_CALL, cfunc)) { \
    x = NULL; \
} \
else { \
    x = call; \
    if (tstate->c_profilefunc != NULL) { \
        if (x == NULL) { \
            call_trace_protected(tstate->c_profilefunc, \
                tstate->c_profileobj, \
                tstate, frame, \
                PyTrace_RETURN, cfunc); \
            /* XXX should pass (type, value, tb) */ \
        } else { \
            if (call_trace(tstate->c_profilefunc, \
                tstate->c_profileobj, \
                tstate, frame, \
                PyTrace_RETURN, cfunc)) { \
                Py_DECREF(x); \
                x = NULL; \
            } \
        } \
    } \
} \

#elif (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION == 10 || PY_MINOR_VERSION == 11)

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/c5bfb88eb6f82111bb1603ae9d78d0476b552d66/Python/ceval.c#L36-L40
 */
typedef struct {
    PyCodeObject *code; // The code object for the bounds. May be NULL.
    PyCodeAddressRange bounds; // Only valid if code != NULL.
    CFrame cframe;
} PyTraceInfo;


/*
 * Code originally from:
 * https://github.com/python/cpython/blob/c5bfb88eb6f82111bb1603ae9d78d0476b552d66/Objects/codeobject.c#L1257-L1266
 * NOTE: The function is renamed.
 */
static void
_nb_PyLineTable_InitAddressRange(const char *linetable, Py_ssize_t length, int firstlineno, PyCodeAddressRange *range)
{
    range->opaque.lo_next = linetable;
    range->opaque.limit = range->opaque.lo_next + length;
    range->ar_start = -1;
    range->ar_end = 0;
    range->opaque.computed_line = firstlineno;
    range->ar_line = -1;
}

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/c5bfb88eb6f82111bb1603ae9d78d0476b552d66/Objects/codeobject.c#L1269-L1275
 * NOTE: The function is renamed.
 */
static int
_nb_PyCode_InitAddressRange(PyCodeObject* co, PyCodeAddressRange *bounds)
{
    const char *linetable = PyBytes_AS_STRING(co->co_linetable);
    Py_ssize_t length = PyBytes_GET_SIZE(co->co_linetable);
    _nb_PyLineTable_InitAddressRange(linetable, length, co->co_firstlineno, bounds);
    return bounds->ar_line;
}

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/c5bfb88eb6f82111bb1603ae9d78d0476b552d66/Python/ceval.c#L5468-L5475
 * NOTE: The call to _PyCode_InitAddressRange is renamed.
 */
static void
initialize_trace_info(PyTraceInfo *trace_info, PyFrameObject *frame)
{
    if (trace_info->code != frame->f_code) {
        trace_info->code = frame->f_code;
        _nb_PyCode_InitAddressRange(frame->f_code, &trace_info->bounds);
    }
}

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/c5bfb88eb6f82111bb1603ae9d78d0476b552d66/Python/ceval.c#L5477-L5501
 */
static int
call_trace(Py_tracefunc func, PyObject *obj,
           PyThreadState *tstate, PyFrameObject *frame,
           PyTraceInfo *trace_info,
           int what, PyObject *arg)
{
    int result;
    if (tstate->tracing)
        return 0;
    tstate->tracing++;
    tstate->cframe->use_tracing = 0;
    if (frame->f_lasti < 0) {
        frame->f_lineno = frame->f_code->co_firstlineno;
    }
    else {
        initialize_trace_info(trace_info, frame);
        frame->f_lineno = _PyCode_CheckLineNumber(frame->f_lasti*sizeof(_Py_CODEUNIT), &trace_info->bounds);
    }
    result = func(obj, frame, what, arg);
    frame->f_lineno = 0;
    tstate->cframe->use_tracing = ((tstate->c_tracefunc != NULL)
                           || (tstate->c_profilefunc != NULL));
    tstate->tracing--;
    return result;
}

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/c5bfb88eb6f82111bb1603ae9d78d0476b552d66/Python/ceval.c#L5445-L5466
 */
static int
call_trace_protected(Py_tracefunc func, PyObject *obj,
                     PyThreadState *tstate, PyFrameObject *frame,
                     PyTraceInfo *trace_info,
                     int what, PyObject *arg)
{
    PyObject *type, *value, *traceback;
    int err;
    PyErr_Fetch(&type, &value, &traceback);
    err = call_trace(func, obj, tstate, frame, trace_info, what, arg);
    if (err == 0)
    {
        PyErr_Restore(type, value, traceback);
        return 0;
    }
    else
    {
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        return -1;
    }
}

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/c5bfb88eb6f82111bb1603ae9d78d0476b552d66/Python/ceval.c#L5810-L5839
 * NOTE: The state test https://github.com/python/cpython/blob/c5bfb88eb6f82111bb1603ae9d78d0476b552d66/Python/ceval.c#L5811
 * has been removed, it's dealt with in call_cfunc.
 */
#define C_TRACE(x, call)                                        \
if (call_trace(tstate->c_profilefunc, tstate->c_profileobj,     \
               tstate, tstate->frame, &trace_info, PyTrace_CALL,\
               cfunc))	                                        \
    x = NULL;                                                   \
else                                                            \
{                                                               \
    x = call;                                                   \
    if (tstate->c_profilefunc != NULL)                          \
    {                                                           \
        if (x == NULL)                                          \
        {                                                       \
            call_trace_protected(tstate->c_profilefunc,         \
                                 tstate->c_profileobj,          \
                                 tstate, tstate->frame,         \
                                 &trace_info,                   \
                                 PyTrace_RETURN, cfunc);	\
            /* XXX should pass (type, value, tb) */             \
        }                                                       \
        else                                                    \
        {                                                       \
            if (call_trace(tstate->c_profilefunc,               \
                           tstate->c_profileobj,                \
                           tstate, tstate->frame,               \
                           &trace_info,                         \
                           PyTrace_RETURN, cfunc))		\
            {                                                   \
                Py_DECREF(x);                                   \
                x = NULL;                                       \
            }                                                   \
        }                                                       \
    }                                                           \
}

#else  // Python <3.10

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/d5650a1738fe34f6e1db4af5f4c4edb7cae90a36/Python/ceval.c#L4242-L4257
 */
static int
call_trace(Py_tracefunc func, PyObject *obj,
           PyThreadState *tstate, PyFrameObject *frame,
           int what, PyObject *arg)
{
    int result;
    if (tstate->tracing)
        return 0;
    tstate->tracing++;
    tstate->use_tracing = 0;
    result = func(obj, frame, what, arg);
    tstate->use_tracing = ((tstate->c_tracefunc != NULL)
                           || (tstate->c_profilefunc != NULL));
    tstate->tracing--;
    return result;
}

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/d5650a1738fe34f6e1db4af5f4c4edb7cae90a36/Python/ceval.c#L4220-L4240
 */
static int
call_trace_protected(Py_tracefunc func, PyObject *obj,
                     PyThreadState *tstate, PyFrameObject *frame,
                     int what, PyObject *arg)
{
    PyObject *type, *value, *traceback;
    int err;
    PyErr_Fetch(&type, &value, &traceback);
    err = call_trace(func, obj, tstate, frame, what, arg);
    if (err == 0)
    {
        PyErr_Restore(type, value, traceback);
        return 0;
    }
    else
    {
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        return -1;
    }
}

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/d5650a1738fe34f6e1db4af5f4c4edb7cae90a36/Python/ceval.c#L4520-L4549
 * NOTE: The state test https://github.com/python/cpython/blob/d5650a1738fe34f6e1db4af5f4c4edb7cae90a36/Python/ceval.c#L4521
 * has been removed, it's dealt with in call_cfunc.
 */
#define C_TRACE(x, call)                                        \
if (call_trace(tstate->c_profilefunc, tstate->c_profileobj,     \
               tstate, tstate->frame, PyTrace_CALL, cfunc))     \
    x = NULL;                                                   \
else                                                            \
{                                                               \
    x = call;                                                   \
    if (tstate->c_profilefunc != NULL)                          \
    {                                                           \
        if (x == NULL)                                          \
        {                                                       \
            call_trace_protected(tstate->c_profilefunc,         \
                                 tstate->c_profileobj,          \
                                 tstate, tstate->frame,         \
                                 PyTrace_RETURN, cfunc);        \
            /* XXX should pass (type, value, tb) */             \
        }                                                       \
        else                                                    \
        {                                                       \
            if (call_trace(tstate->c_profilefunc,               \
                           tstate->c_profileobj,                \
                           tstate, tstate->frame,               \
                           PyTrace_RETURN, cfunc))              \
            {                                                   \
                Py_DECREF(x);                                   \
                x = NULL;                                       \
            }                                                   \
        }                                                       \
    }                                                           \
}


#endif

typedef std::vector<Type> TypeTable;
typedef std::vector<PyObject*> Functions;

/* The Dispatcher class is the base class of all dispatchers in the CPU and
   CUDA targets. Its main responsibilities are:

   - Resolving the best overload to call for a given set of arguments, and
   - Calling the resolved overload.

   This logic is implemented within this class for efficiency (lookup of the
   appropriate overload needs to be fast) and ease of implementation (calling
   directly into a compiled function using a function pointer is easier within
   the C++ code where the overload has been resolved). */
class Dispatcher {
public:
    PyObject_HEAD
    /* Whether compilation of new overloads is permitted */
    char can_compile;
    /* Whether fallback to object mode is permitted */
    char can_fallback;
    /* Whether types must match exactly when resolving overloads.
       If not, conversions (e.g. float32 -> float64) are permitted when
       searching for a match. */
    char exact_match_required;
    /* Borrowed reference */
    PyObject *fallbackdef;
    /* Whether to fold named arguments and default values
      (false for lifted loops) */
    int fold_args;
    /* Whether the last positional argument is a stararg */
    int has_stararg;
    /* Tuple of argument names */
    PyObject *argnames;
    /* Tuple of default values */
    PyObject *defargs;
    /* Number of arguments to function */
    int argct;
    /* Used for selecting overloaded function implementations */
    TypeManager *tm;
    /* An array of overloads */
    Functions functions;
    /* A flattened array of argument types to all overloads
     * (invariant: sizeof(overloads) == argct * sizeof(functions)) */
    TypeTable overloads;

    /* Add a new overload. Parameters:

       - args: An array of Type objects, one for each parameter
       - callable: The callable implementing this overload. */
    void addDefinition(Type args[], PyObject *callable) {
        overloads.reserve(argct + overloads.size());
        for (int i=0; i<argct; ++i) {
            overloads.push_back(args[i]);
        }
        functions.push_back(callable);
    }

    /* Given a list of types, find the overloads that have a matching signature.
       Returns the best match, as well as the number of matches found.

       Parameters:

       - sig: an array of Type objects, one for each parameter.
       - matches: the number of matches found (mutated by this function).
       - allow_unsafe: whether to match overloads that would require an unsafe
                       cast.
       - exact_match_required: Whether all arguments types must match the
                               overload's types exactly. When false,
                               overloads that would require a type conversion
                               can also be matched. */
    PyObject* resolve(Type sig[], int &matches, bool allow_unsafe,
                      bool exact_match_required) const {
        const int ovct = functions.size();
        int selected;
        matches = 0;
        if (0 == ovct) {
            // No overloads registered
            return NULL;
        }
        if (argct == 0) {
            // Nullary function: trivial match on first overload
            matches = 1;
            selected = 0;
        }
        else {
            matches = tm->selectOverload(sig, &overloads[0], selected, argct,
                                         ovct, allow_unsafe,
                                         exact_match_required);
        }
        if (matches == 1) {
            return functions[selected];
        }
        return NULL;
    }

    /* Remove all overloads */
    void clear() {
        functions.clear();
        overloads.clear();
    }

};


static int
Dispatcher_traverse(Dispatcher *self, visitproc visit, void *arg)
{
    Py_VISIT(self->defargs);
    return 0;
}

static void
Dispatcher_dealloc(Dispatcher *self)
{
    Py_XDECREF(self->argnames);
    Py_XDECREF(self->defargs);
    self->clear();
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static int
Dispatcher_init(Dispatcher *self, PyObject *args, PyObject *kwds)
{
    PyObject *tmaddrobj;
    void *tmaddr;
    int argct;
    int can_fallback;
    int has_stararg = 0;
    int exact_match_required = 0;

    if (!PyArg_ParseTuple(args, "OiiO!O!i|ii", &tmaddrobj, &argct,
                          &self->fold_args,
                          &PyTuple_Type, &self->argnames,
                          &PyTuple_Type, &self->defargs,
                          &can_fallback,
                          &has_stararg,
                          &exact_match_required
                         )) {
        return -1;
    }
    Py_INCREF(self->argnames);
    Py_INCREF(self->defargs);
    tmaddr = PyLong_AsVoidPtr(tmaddrobj);
    self->tm = static_cast<TypeManager*>(tmaddr);
    self->argct = argct;
    self->can_compile = 1;
    self->can_fallback = can_fallback;
    self->fallbackdef = NULL;
    self->has_stararg = has_stararg;
    self->exact_match_required = exact_match_required;
    return 0;
}

static PyObject *
Dispatcher_clear(Dispatcher *self, PyObject *args)
{
    self->clear();
    Py_RETURN_NONE;
}

static
PyObject*
Dispatcher_Insert(Dispatcher *self, PyObject *args, PyObject *kwds)
{
    /* The cuda kwarg is a temporary addition until CUDA overloads are compiled
     * functions. Once they are compiled functions, kwargs can be removed from
     * this function. */
    static char *keywords[] = {
        (char*)"sig",
        (char*)"func",
        (char*)"objectmode",
        (char*)"cuda",
        NULL
    };

    PyObject *sigtup, *cfunc;
    int i, sigsz;
    int *sig;
    int objectmode = 0;
    int cuda = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|ip", keywords, &sigtup,
                                     &cfunc, &objectmode, &cuda)) {
        return NULL;
    }

    if (!cuda && !PyObject_TypeCheck(cfunc, &PyCFunction_Type) ) {
        PyErr_SetString(PyExc_TypeError, "must be builtin_function_or_method");
        return NULL;
    }

    sigsz = PySequence_Fast_GET_SIZE(sigtup);
    sig = new int[sigsz];

    for (i = 0; i < sigsz; ++i) {
        sig[i] = PyLong_AsLong(PySequence_Fast_GET_ITEM(sigtup, i));
    }

    /* The reference to cfunc is borrowed; this only works because the
       derived Python class also stores an (owned) reference to cfunc. */
    self->addDefinition(sig, cfunc);

    /* Add pure python fallback */
    if (!self->fallbackdef && objectmode){
        self->fallbackdef = cfunc;
    }

    delete[] sig;

    Py_RETURN_NONE;
}

static
void explain_issue(PyObject *dispatcher, PyObject *args, PyObject *kws,
                   const char *method_name, const char *default_msg)
{
    PyObject *callback, *result;
    callback = PyObject_GetAttrString(dispatcher, method_name);
    if (!callback) {
        PyErr_SetString(PyExc_TypeError, default_msg);
        return;
    }
    result = PyObject_Call(callback, args, kws);
    Py_DECREF(callback);
    if (result != NULL) {
        PyErr_Format(PyExc_RuntimeError, "%s must raise an exception",
                     method_name);
        Py_DECREF(result);
    }
}

static
void explain_ambiguous(PyObject *dispatcher, PyObject *args, PyObject *kws)
{
    explain_issue(dispatcher, args, kws, "_explain_ambiguous",
                  "Ambiguous overloading");
}

static
void explain_matching_error(PyObject *dispatcher, PyObject *args, PyObject *kws)
{
    explain_issue(dispatcher, args, kws, "_explain_matching_error",
                  "No matching definition");
}

static
int search_new_conversions(PyObject *dispatcher, PyObject *args, PyObject *kws)
{
    PyObject *callback, *result;
    int res;

    callback = PyObject_GetAttrString(dispatcher,
                                      "_search_new_conversions");
    if (!callback) {
        return -1;
    }
    result = PyObject_Call(callback, args, kws);
    Py_DECREF(callback);
    if (result == NULL) {
        return -1;
    }
    if (!PyBool_Check(result)) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError,
                        "_search_new_conversions() should return a boolean");
        return -1;
    }
    res = (result == Py_True) ? 1 : 0;
    Py_DECREF(result);
    return res;
}


/* A custom, fast, inlinable version of PyCFunction_Call() */
static PyObject *
call_cfunc(Dispatcher *self, PyObject *cfunc, PyObject *args, PyObject *kws, PyObject *locals)
{
    PyCFunctionWithKeywords fn;
    PyThreadState *tstate;

    assert(PyCFunction_Check(cfunc));
    assert(PyCFunction_GET_FLAGS(cfunc) == (METH_VARARGS | METH_KEYWORDS));
    fn = (PyCFunctionWithKeywords) PyCFunction_GET_FUNCTION(cfunc);
    tstate = PyThreadState_GET();

#if (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION == 12)
    /* FIXME: as per the changes from PEP 669 Numba does not support profiling
     * (yet?). Just skip frame injection entirely. JIT compiled functions will
     * NOT show up in cProfile profiler.
     */
    if (0)
#elif (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION == 11)
    /*
     * On Python 3.11, _PyEval_EvalFrameDefault stops using PyTraceInfo since 
     * it's now baked into ThreadState.
     * https://github.com/python/cpython/pull/26623
     */
    if (tstate->cframe->use_tracing && tstate->c_profilefunc)
#elif (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION == 10)
    /*
     * On Python 3.10+ trace_info comes from somewhere up in PyFrameEval et al,
     * Numba doesn't have access to that so creates an equivalent struct and
     * wires it up against the cframes. This is passed into the tracing
     * functions.
     *
     * Code originally from:
     * https://github.com/python/cpython/blob/c5bfb88eb6f82111bb1603ae9d78d0476b552d66/Python/ceval.c#L1611-L1622
     */
    PyTraceInfo trace_info;
    trace_info.code = NULL; // not initialized
    CFrame *prev_cframe = tstate->cframe;
    trace_info.cframe.use_tracing = prev_cframe->use_tracing;
    trace_info.cframe.previous = prev_cframe;

    if (trace_info.cframe.use_tracing && tstate->c_profilefunc)
#else
    /*
     * On Python prior to 3.10, tracing state is a member of the threadstate
     */
    if (tstate->use_tracing && tstate->c_profilefunc)
#endif
    {
        /*
         * The following code requires some explaining:
         *
         * We want the jit-compiled function to be visible to the profiler, so we
         * need to synthesize a frame for it.
         * The PyFrame_New() constructor doesn't do anything with the 'locals' value if the 'code's
         * 'CO_NEWLOCALS' flag is set (which is always the case nowadays).
         * So, to get local variables into the frame, we have to manually set the 'f_locals'
         * member, then call `PyFrame_LocalsToFast`, where a subsequent call to the `frame.f_locals`
         * property (by virtue of the `frame_getlocals` function in frameobject.c) will find them.
         */
        PyCodeObject *code = (PyCodeObject*)PyObject_GetAttrString((PyObject*)self, "__code__");
        PyObject *globals = PyDict_New();
        PyObject *builtins = PyEval_GetBuiltins();
        PyFrameObject *frame = NULL;
        PyObject *result = NULL;

        if (!code) {
            PyErr_Format(PyExc_RuntimeError, "No __code__ attribute found.");
            goto error;
        }
        /* Populate builtins, which is required by some JITted functions */
        if (PyDict_SetItemString(globals, "__builtins__", builtins)) {
            goto error;
        }

        /* unset the CO_OPTIMIZED flag, make the frame get a new locals dict */
        code->co_flags &= 0xFFFE;

        frame = PyFrame_New(tstate, code, globals, locals);
        if (frame == NULL) {
            goto error;
        }
        /* Populate the 'fast locals' in `frame` */
        PyFrame_LocalsToFast(frame, 0);
#if (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION == 12)
        /* empty */
#elif (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION == 11)
        C_TRACE(result, fn(PyCFunction_GET_SELF(cfunc), args, kws), frame);
#else
        tstate->frame = frame;
        C_TRACE(result, fn(PyCFunction_GET_SELF(cfunc), args, kws));
        /* write changes back to locals? */
        PyFrame_FastToLocals(frame);
        tstate->frame = frame->f_back;
#endif
    error:
        Py_XDECREF(frame);
        Py_XDECREF(globals);
        Py_XDECREF(code);
        return result;
    }
    else
    {
        return fn(PyCFunction_GET_SELF(cfunc), args, kws);
    }
}

static
PyObject*
compile_and_invoke(Dispatcher *self, PyObject *args, PyObject *kws, PyObject *locals)
{
    /* Compile a new one */
    PyObject *cfa, *cfunc, *retval;
    cfa = PyObject_GetAttrString((PyObject*)self, "_compile_for_args");
    if (cfa == NULL)
        return NULL;

    /* NOTE: we call the compiled function ourselves instead of
       letting the Python derived class do it.  This is for proper
       behaviour of globals() in jitted functions (issue #476). */
    cfunc = PyObject_Call(cfa, args, kws);
    Py_DECREF(cfa);

    if (cfunc == NULL)
        return NULL;

    if (PyObject_TypeCheck(cfunc, &PyCFunction_Type)) {
        retval = call_cfunc(self, cfunc, args, kws, locals);
    } else {
        /* Re-enter interpreter */
        retval = PyObject_Call(cfunc, args, kws);
    }
    Py_DECREF(cfunc);

    return retval;
}

/* A copy of compile_and_invoke, that only compiles. This is needed for CUDA
 * kernels, because its overloads are Python instances of the _Kernel class,
 * rather than compiled functions. Once CUDA overloads are compiled functions,
 * cuda_compile_only can be removed. */
static
PyObject*
cuda_compile_only(Dispatcher *self, PyObject *args, PyObject *kws, PyObject *locals)
{
    /* Compile a new one */
    PyObject *cfa, *cfunc;
    cfa = PyObject_GetAttrString((PyObject*)self, "_compile_for_args");
    if (cfa == NULL)
        return NULL;

    cfunc = PyObject_Call(cfa, args, kws);
    Py_DECREF(cfa);

    return cfunc;
}

static int
find_named_args(Dispatcher *self, PyObject **pargs, PyObject **pkws)
{
    PyObject *oldargs = *pargs, *newargs;
    PyObject *kws = *pkws;
    Py_ssize_t pos_args = PyTuple_GET_SIZE(oldargs);
    Py_ssize_t named_args, total_args, i;
    Py_ssize_t func_args = PyTuple_GET_SIZE(self->argnames);
    Py_ssize_t defaults = PyTuple_GET_SIZE(self->defargs);
    /* Last parameter with a default value */
    Py_ssize_t last_def = (self->has_stararg)
                          ? func_args - 2
                          : func_args - 1;
    /* First parameter with a default value */
    Py_ssize_t first_def = last_def - defaults + 1;
    /* Minimum number of required arguments */
    Py_ssize_t minargs = first_def;

    if (kws != NULL)
        named_args = PyDict_Size(kws);
    else
        named_args = 0;
    total_args = pos_args + named_args;
    if (!self->has_stararg && total_args > func_args) {
        PyErr_Format(PyExc_TypeError,
                     "too many arguments: expected %d, got %d",
                     (int) func_args, (int) total_args);
        return -1;
    }
    else if (total_args < minargs) {
        if (minargs == func_args)
            PyErr_Format(PyExc_TypeError,
                         "not enough arguments: expected %d, got %d",
                         (int) minargs, (int) total_args);
        else
            PyErr_Format(PyExc_TypeError,
                         "not enough arguments: expected at least %d, got %d",
                         (int) minargs, (int) total_args);
        return -1;
    }
    newargs = PyTuple_New(func_args);
    if (!newargs)
        return -1;
    /* First pack the stararg */
    if (self->has_stararg) {
        Py_ssize_t stararg_size = Py_MAX(0, pos_args - func_args + 1);
        PyObject *stararg = PyTuple_New(stararg_size);
        if (!stararg) {
            Py_DECREF(newargs);
            return -1;
        }
        for (i = 0; i < stararg_size; i++) {
            PyObject *value = PyTuple_GET_ITEM(oldargs, func_args - 1 + i);
            Py_INCREF(value);
            PyTuple_SET_ITEM(stararg, i, value);
        }
        /* Put it in last position */
        PyTuple_SET_ITEM(newargs, func_args - 1, stararg);

    }
    for (i = 0; i < pos_args; i++) {
        PyObject *value = PyTuple_GET_ITEM(oldargs, i);
        if (self->has_stararg && i >= func_args - 1) {
            /* Skip stararg */
            break;
        }
        Py_INCREF(value);
        PyTuple_SET_ITEM(newargs, i, value);
    }

    /* Iterate over missing positional arguments, try to find them in
       named arguments or default values. */
    for (i = pos_args; i < func_args; i++) {
        PyObject *name = PyTuple_GET_ITEM(self->argnames, i);
        if (self->has_stararg && i >= func_args - 1) {
            /* Skip stararg */
            break;
        }
        if (kws != NULL) {
            /* Named argument? */
            PyObject *value = PyDict_GetItem(kws, name);
            if (value != NULL) {
                Py_INCREF(value);
                PyTuple_SET_ITEM(newargs, i, value);
                named_args--;
                continue;
            }
        }
        if (i >= first_def && i <= last_def) {
            /* Argument has a default value? */
            PyObject *value = PyTuple_GET_ITEM(self->defargs, i - first_def);
            Py_INCREF(value);
            PyTuple_SET_ITEM(newargs, i, value);
            continue;
        }
        else if (i < func_args - 1 || !self->has_stararg) {
            PyErr_Format(PyExc_TypeError,
                         "missing argument '%s'",
                         PyString_AsString(name));
            Py_DECREF(newargs);
            return -1;
        }
    }
    if (named_args) {
        PyErr_Format(PyExc_TypeError,
                     "some keyword arguments unexpected");
        Py_DECREF(newargs);
        return -1;
    }
    *pargs = newargs;
    *pkws = NULL;
    return 0;
}


/*
 * Management of thread-local
 */

#ifdef _MSC_VER
#define THREAD_LOCAL(ty) __declspec(thread) ty
#else
/* Non-standard C99 extension that's understood by gcc and clang */
#define THREAD_LOCAL(ty) __thread ty
#endif

static THREAD_LOCAL(bool) use_tls_target_stack;


struct raii_use_tls_target_stack {
    bool old_setting;

    raii_use_tls_target_stack(bool new_setting)
        : old_setting(use_tls_target_stack)
    {
        use_tls_target_stack = new_setting;
    }

    ~raii_use_tls_target_stack() {
        use_tls_target_stack = old_setting;
    }
};

static PyObject*
Dispatcher_call(Dispatcher *self, PyObject *args, PyObject *kws)
{
    PyObject *tmptype, *retval = NULL;
    int *tys = NULL;
    int argct;
    int i;
    int prealloc[24];
    int matches;
    PyObject *cfunc;
    PyThreadState *ts = PyThreadState_Get();
    PyObject *locals = NULL;

    // Check TLS target stack
    if (use_tls_target_stack) {
        raii_use_tls_target_stack turn_off(false);
        PyObject * meth_call_tls_target;
        meth_call_tls_target = PyObject_GetAttrString((PyObject*)self,
                                                      "_call_tls_target");
        if (!meth_call_tls_target) return NULL;
        // Transfer control to self._call_tls_target
        retval = PyObject_Call(meth_call_tls_target, args, kws);
        Py_DECREF(meth_call_tls_target);
        return retval;
    }

    /* If compilation is enabled, ensure that an exact match is found and if
     * not compile one */
    int exact_match_required = self->can_compile ? 1 : self->exact_match_required;

#if (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION >= 10)
    if (ts->tracing && ts->c_profilefunc) {
#else
    if (ts->use_tracing && ts->c_profilefunc) {
#endif
        locals = PyEval_GetLocals();
        if (locals == NULL) {
            goto CLEANUP;
        }
    }
    if (self->fold_args) {
        if (find_named_args(self, &args, &kws))
            return NULL;
    }
    else
        Py_INCREF(args);
    /* Now we own a reference to args */

    argct = PySequence_Fast_GET_SIZE(args);

    if (argct < (Py_ssize_t) (sizeof(prealloc) / sizeof(int)))
        tys = prealloc;
    else
        tys = new int[argct];

    for (i = 0; i < argct; ++i) {
        tmptype = PySequence_Fast_GET_ITEM(args, i);
        tys[i] = typeof_typecode((PyObject *) self, tmptype);
        if (tys[i] == -1) {
            if (self->can_fallback){
                /* We will clear the exception if fallback is allowed. */
                PyErr_Clear();
            } else {
                goto CLEANUP;
            }
        }
    }

    /* We only allow unsafe conversions if compilation of new specializations
       has been disabled.

       Note that the number of matches is returned in matches by resolve, which
       accepts it as a reference. */
    cfunc = self->resolve(tys, matches, !self->can_compile,
                          exact_match_required);

    if (matches == 0 && !self->can_compile) {
        /*
         * If we can't compile a new specialization, look for
         * matching signatures for which conversions haven't been
         * registered on the C++ TypeManager.
         */
        int res = search_new_conversions((PyObject *) self, args, kws);
        if (res < 0) {
            retval = NULL;
            goto CLEANUP;
        }
        if (res > 0) {
            /* Retry with the newly registered conversions */
            cfunc = self->resolve(tys, matches, !self->can_compile,
                                  exact_match_required);
        }
    }
    if (matches == 1) {
        /* Definition is found */
        retval = call_cfunc(self, cfunc, args, kws, locals);
    } else if (matches == 0) {
        /* No matching definition */
        if (self->can_compile) {
            retval = compile_and_invoke(self, args, kws, locals);
        } else if (self->fallbackdef) {
            /* Have object fallback */
            retval = call_cfunc(self, self->fallbackdef, args, kws, locals);
        } else {
            /* Raise TypeError */
            explain_matching_error((PyObject *) self, args, kws);
            retval = NULL;
        }
    } else if (self->can_compile) {
        /* Ambiguous, but are allowed to compile */
        retval = compile_and_invoke(self, args, kws, locals);
    } else {
        /* Ambiguous */
        explain_ambiguous((PyObject *) self, args, kws);
        retval = NULL;
    }

CLEANUP:
    if (tys != prealloc)
        delete[] tys;
    Py_DECREF(args);

    return retval;
}

/* Based on Dispatcher_call above, with the following differences:
   1. It does not invoke the definition of the function.
   2. It returns the definition, instead of a value returned by the function.

   This is because CUDA functions are, at present, _Kernel objects rather than
   compiled functions. */
static PyObject*
Dispatcher_cuda_call(Dispatcher *self, PyObject *args, PyObject *kws)
{
    PyObject *tmptype, *retval = NULL;
    int *tys = NULL;
    int argct;
    int i;
    int prealloc[24];
    int matches;
    PyObject *cfunc;
    PyThreadState *ts = PyThreadState_Get();
    PyObject *locals = NULL;

    /* If compilation is enabled, ensure that an exact match is found and if
     * not compile one */
    int exact_match_required = self->can_compile ? 1 : self->exact_match_required;

#if (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION >= 10)
    if (ts->tracing && ts->c_profilefunc) {
#else
    if (ts->use_tracing && ts->c_profilefunc) {
#endif
        locals = PyEval_GetLocals();
        if (locals == NULL) {
            goto CLEANUP;
        }
    }
    if (self->fold_args) {
        if (find_named_args(self, &args, &kws))
            return NULL;
    }
    else
        Py_INCREF(args);
    /* Now we own a reference to args */

    argct = PySequence_Fast_GET_SIZE(args);

    if (argct < (Py_ssize_t) (sizeof(prealloc) / sizeof(int)))
        tys = prealloc;
    else
        tys = new int[argct];

    for (i = 0; i < argct; ++i) {
        tmptype = PySequence_Fast_GET_ITEM(args, i);
        tys[i] = typeof_typecode((PyObject *) self, tmptype);
        if (tys[i] == -1) {
            if (self->can_fallback){
                /* We will clear the exception if fallback is allowed. */
                PyErr_Clear();
            } else {
                goto CLEANUP;
            }
        }
    }

    /* We only allow unsafe conversions if compilation of new specializations
       has been disabled. */
    cfunc = self->resolve(tys, matches, !self->can_compile,
                          exact_match_required);

    if (matches == 0 && !self->can_compile) {
        /*
         * If we can't compile a new specialization, look for
         * matching signatures for which conversions haven't been
         * registered on the C++ TypeManager.
         */
        int res = search_new_conversions((PyObject *) self, args, kws);
        if (res < 0) {
            retval = NULL;
            goto CLEANUP;
        }
        if (res > 0) {
            /* Retry with the newly registered conversions */
            cfunc = self->resolve(tys, matches, !self->can_compile,
                                  exact_match_required);
        }
    }

    if (matches == 1) {
        /* Definition is found */
        retval = cfunc;
        Py_INCREF(retval);
    } else if (matches == 0) {
        /* No matching definition */
        if (self->can_compile) {
            retval = cuda_compile_only(self, args, kws, locals);
        } else if (self->fallbackdef) {
            /* Have object fallback */
            retval = call_cfunc(self, self->fallbackdef, args, kws, locals);
        } else {
            /* Raise TypeError */
            explain_matching_error((PyObject *) self, args, kws);
            retval = NULL;
        }
    } else if (self->can_compile) {
        /* Ambiguous, but are allowed to compile */
        retval = cuda_compile_only(self, args, kws, locals);
    } else {
        /* Ambiguous */
        explain_ambiguous((PyObject *) self, args, kws);
        retval = NULL;
    }

CLEANUP:
    if (tys != prealloc)
        delete[] tys;
    Py_DECREF(args);

    return retval;
}

static int
import_devicearray(void)
{
    PyObject *devicearray = PyImport_ImportModule("numba._devicearray");
    if (devicearray == NULL) {
        return -1;
    }
    Py_DECREF(devicearray);

    DeviceArray_API = (void**)PyCapsule_Import("numba._devicearray._DEVICEARRAY_API", 0);
    if (DeviceArray_API == NULL) {
        return -1;
    }

    return 0;
}

static PyMethodDef Dispatcher_methods[] = {
    { "_clear", (PyCFunction)Dispatcher_clear, METH_NOARGS, NULL },
    { "_insert", (PyCFunction)Dispatcher_Insert, METH_VARARGS | METH_KEYWORDS,
      "insert new definition"},
    { "_cuda_call", (PyCFunction)Dispatcher_cuda_call,
      METH_VARARGS | METH_KEYWORDS, "CUDA call resolution" },
    { NULL },
};

static PyMemberDef Dispatcher_members[] = {
    {(char*)"_can_compile", T_BOOL, offsetof(Dispatcher, can_compile), 0, NULL },
    {NULL}  /* Sentinel */
};


static PyTypeObject DispatcherType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_dispatcher.Dispatcher",                    /* tp_name */
    sizeof(Dispatcher),                          /* tp_basicsize */
    0,                                           /* tp_itemsize */
    (destructor)Dispatcher_dealloc,              /* tp_dealloc */
    0,                                           /* tp_vectorcall_offset */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    0,                                           /* tp_as_async */
    0,                                           /* tp_repr */
    0,                                           /* tp_as_number */
    0,                                           /* tp_as_sequence */
    0,                                           /* tp_as_mapping */
    0,                                           /* tp_hash */
    (PyCFunctionWithKeywords)Dispatcher_call,    /* tp_call*/
    0,                                           /* tp_str*/
    0,                                           /* tp_getattro*/
    0,                                           /* tp_setattro*/
    0,                                           /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags*/
    "Dispatcher object",                         /* tp_doc */
    (traverseproc) Dispatcher_traverse,          /* tp_traverse */
    0,                                           /* tp_clear */
    0,                                           /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    0,                                           /* tp_iter */
    0,                                           /* tp_iternext */
    Dispatcher_methods,                          /* tp_methods */
    Dispatcher_members,                          /* tp_members */
    0,                                           /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    (initproc)Dispatcher_init,                   /* tp_init */
    0,                                           /* tp_alloc */
    0,                                           /* tp_new */
    0,                                           /* tp_free */
    0,                                           /* tp_is_gc */
    0,                                           /* tp_bases */
    0,                                           /* tp_mro */
    0,                                           /* tp_cache */
    0,                                           /* tp_subclasses */
    0,                                           /* tp_weaklist */
    0,                                           /* tp_del */
    0,                                           /* tp_version_tag */
    0,                                           /* tp_finalize */
    0,                                           /* tp_vectorcall */
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 12)
/* This was introduced first in 3.12
 * https://github.com/python/cpython/issues/91051
 */
    0,                                           /* tp_watched */
#endif

/* WARNING: Do not remove this, only modify it! It is a version guard to
 * act as a reminder to update this struct on Python version update! */
#if (PY_MAJOR_VERSION == 3)
#if ! ((PY_MINOR_VERSION == 9) || (PY_MINOR_VERSION == 10) || (PY_MINOR_VERSION == 11) || (PY_MINOR_VERSION == 12))
#error "Python minor version is not supported."
#endif
#else
#error "Python major version is not supported."
#endif
/* END WARNING*/
};


static PyObject *compute_fingerprint(PyObject *self, PyObject *args)
{
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O:compute_fingerprint", &val))
        return NULL;
    return typeof_compute_fingerprint(val);
}

static PyObject *set_use_tls_target_stack(PyObject *self, PyObject *args)
{
    int val;
    if (!PyArg_ParseTuple(args, "p", &val))
        return NULL;
    bool old = use_tls_target_stack;
    use_tls_target_stack = val;
    // return the old value
    if (old) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(typeof_init),
    declmethod(compute_fingerprint),
    declmethod(set_use_tls_target_stack),
    { NULL },
#undef declmethod
};


MOD_INIT(_dispatcher) {
    if (import_devicearray() < 0) {
      PyErr_Print();
      PyErr_SetString(PyExc_ImportError, "numba._devicearray failed to import");
      return MOD_ERROR_VAL;
    }

    PyObject *m;
    MOD_DEF(m, "_dispatcher", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    DispatcherType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&DispatcherType) < 0) {
        return MOD_ERROR_VAL;
    }
    Py_INCREF(&DispatcherType);
    PyModule_AddObject(m, "Dispatcher", (PyObject*)(&DispatcherType));

    return MOD_SUCCESS_VAL(m);
}
