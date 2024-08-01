/*
This is a copy of the sys monitoring logic in _dispatcher.cpp for Python 3.12.
This is necessary to workaround issue with g++ complaining about C++ template
inside "internal/pycore_mimalloc.h" from #include "internal/pycore_interp.h" 
*/
#include "_pymodule.h"

#include <string.h>
#include <time.h>
#include <assert.h>

#include "_typeof.h"
#include "frameobject.h"
#include "traceback.h"


#ifndef Py_BUILD_CORE
    #define Py_BUILD_CORE 1
#endif
#include "internal/pycore_frame.h"
// This is a fix suggested in the comments in https://github.com/python/cpython/issues/108216
// specifically https://github.com/python/cpython/issues/108216#issuecomment-1696565797
#ifdef HAVE_STD_ATOMIC
#  undef HAVE_STD_ATOMIC
#endif
#undef _PyGC_FINALIZED
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 12) 
    #include "internal/pycore_atomic.h"
#endif
#include "internal/pycore_interp.h"
#include "internal/pycore_pyerrors.h"
#include "internal/pycore_instruments.h"
#include "internal/pycore_call.h"
#include "cpython/code.h"

// Python 3.12 has a completely new approach to tracing and profiling due to
// the new `sys.monitoring` system.

// From: https://github.com/python/cpython/blob/0ab2384c5f56625e99bb35417cadddfe24d347e1/Python/instrumentation.c#L863-L868
static const int8_t MOST_SIG_BIT[16] = {-1, 0, 1, 1,
                                         2, 2, 2, 2,
                                         3, 3, 3, 3,
                                         3, 3, 3, 3};

// From: https://github.com/python/cpython/blob/0ab2384c5f56625e99bb35417cadddfe24d347e1/Python/instrumentation.c#L873-L879

static inline int msb(uint8_t bits) {
    if (bits > 15) {
        return MOST_SIG_BIT[bits>>4]+4;
    }
    return MOST_SIG_BIT[bits];
}

typedef PyObject Dispatcher;

static int invoke_monitoring(PyThreadState * tstate, int event, Dispatcher *self, PyObject* retval)
{
    // This will invoke monitoring tools (if present) for the event `event`.
    //
    // Arguments:
    //   tstate - the interpreter thread state
    //   event - an event as defined in internal/pycore_instruments.h
    //   self - the dispatcher
    //   retval - the return value from running the dispatcher machine code (if needed)
    //            or NULL if not needed.
    //
    // Return:
    // status 0 for success -1 otherwise.
    //
    // Notes:
    // Python 3.12 has a new monitoring system as described in PEP 669. It's
    // largely implemented in CPython PR #103083.
    //
    // This PEP manifests as a set of monitoring instrumentation in the form of
    // per-monitoring-tool-type callbacks stored as part of the interpreter
    // state (can also be on the code object for "local events" but Numba
    // doesn't support those, see the Numba developer docs). From the Python
    // interpreter this appears as `sys.monitoring`, from the C-side there's not
    // a great deal of public API for the sort of things that Numba wants/needs
    // to do.
    //
    // The new monitoring system is event based, the general idea in the
    // following code is to see if a monitoring "tool" has registered a callback
    // to run on the presence of a particular event and run those callbacks if
    // so. In Numba's case we're just about to disappear into machine code
    // that's essentially doing the same thing as the interpreter would if it
    // executed the bytecode present in the function that's been JIT compiled.
    // As a result we need to tell any tool that has a callback registered for a
    // PY_MONITORING_EVENT_PY_START that a Python function is about to start
    // (and do something similar for when a function returns/raises).
    // This is a total lie as the execution is in machine code, but telling this
    // lie makes it look like a python function has started executing at the
    // point the machine code function starts and tools like profilers will be
    // able to identify this and do something appropriate. The "lie" is very
    // much like lie told for Python < 3.12, but the format of the lie is
    // different. There is no fake frame involved, it's just about calling an
    // appropriate call back, which in a way is a lot less confusing to deal
    // with.
    //
    // For reference, under cProfile all these are NULL, don't even look at
    // them, they are legacy, you need to use the monitoring system!
    // tstate->c_profilefunc
    // tstate->c_profileobj
    // tstate->c_tracefunc
    // tstate->c_traceobj
    //
    // Finally: Useful places to look in the CPython code base:
    // 1. internal/pycore_instruments.h which has the #defines for all the event
    // types and the "types" of tools e.g. debugger, profiler.
    // 2. Python/instrumentation.c which is where most of the implementation is
    // done. Particularly functions `call_instrumentation_vector` and
    // `call_one_instrument`.
    // Note that Python/legacy_tracing.c is not somewhere to look, it's just
    // wiring old style tracing that has been setup via e.g. C-API
    // PyEval_SetProfile into the new monitoring system.
    //
    // Other things...
    // 1. Calls to `sys.monitoring.set_events` clobber the previous state.
    // 2. You can register callbacks for an event without having the event set.
    // 3. You can set events and have no associated callback.
    // 4. Tools are supposed to be respectful of other tools that are
    //    registered, i.e. not clobber/interfere with each other.
    // 5. There are multiple slots for tools, cProfile is a profiler and
    //    profilers should register in slot 2 by convention.
    //
    // This is useful for debug:
    // To detect whether Python is doing _any_ monitoring it's necessary to
    // inspect the per-thread state interpreter monitors.tools member, its a
    // uchar[15]. A non-zero value in any tools slot suggests something
    // is registered to be called on the occurence of some event.
    //
    // bool monitoring_tools_present = false;
    // for (int i = 0; i < _PY_MONITORING_UNGROUPED_EVENTS; i++) {
    //     if (tstate->interp->monitors.tools[i]) {
    //         monitoring_tools_present = true;
    //         break;
    //     }
    // }

    // The code in this function is based loosely on a combination of the
    // following:
    // https://github.com/python/cpython/blob/0ab2384c5f56625e99bb35417cadddfe24d347e1/Python/instrumentation.c#L945-L1008
    // https://github.com/python/cpython/blob/0ab2384c5f56625e99bb35417cadddfe24d347e1/Python/instrumentation.c#L1010-L1026
    // https://github.com/python/cpython/blob/0ab2384c5f56625e99bb35417cadddfe24d347e1/Python/instrumentation.c#L839-L861

    // TODO: check this, call_instrumentation_vector has this at the top.
    if (tstate->tracing){
        return 0;
    }

    // Are there any tools set on this thead for this event?
    uint8_t tools = tstate->interp->monitors.tools[event];
    // offset value for use in callbacks
    PyObject * offset_obj = NULL;
    // callback args slots (used in vectorcall protocol)
    PyObject * callback_args[3] = {NULL, NULL, NULL};

    // If so...
    if (tools)
    {


        PyObject *result = NULL;
        PyCodeObject *code = (PyCodeObject*)PyObject_GetAttrString((PyObject*)self, "__code__"); // incref code
        if (!code) {
            PyErr_Format(PyExc_RuntimeError, "No __code__ attribute found.");
            return -1;
        }

        // TODO: handle local events, see `get_tools_for_instruction`.
        // The issue with local events is that they maybe don't make a lot of
        // sense in a JIT context. The way it works is that
        // `sys.monitoring.set_local_events` takes the code object of a function
        // and "instruments" it with respect to the requested events. In
        // practice this seems to materialise as swapping bytecodes associated
        // with the event bitmask for `INSTRUMENTED_` variants of those
        // bytecodes. Then at interpretation time if an instrumented instruction
        // is encountered it triggers lookups in the `code->_co_monitoring`
        // struct for tools and active monitors etc. In Numba we _know_ the
        // bytecode at which the code starts and we can probably scrape the code
        // to look for instrumented return instructions, so it is feasible to
        // support at least PY_START and PY_RETURN events, however, it's a lot
        // of effort for perhaps something that's practically not that useful.
        // As a result, only global events are supported at present.

        // This is supposed to be the offset of the
        // currently-being-interpreted bytecode instruction. In Numba's case
        // there is no bytecode executing. We know that for a PY_START event
        // that the offset is probably zero (it might be 2 if there's a
        // closure, it's whereever the `RESUME` bytecode appears). However,
        // we don't know which bytecode will be associated with the return
        // (without huge effort to wire that through to here). Therefore
        // zero is also used for return/raise/unwind, the main use case,
        // cProfile, seems to manage to do something sensible even though this
        // is inaccurate.
        offset_obj = PyLong_FromSsize_t(0); // incref offset_obj

        // This is adapted from call_one_instrument. Note that Numba has to care
        // about all events even though it only emits fake events for PY_START,
        // PY_RETURN, RAISE and PY_UNWIND, this is because of the ability of
        // `objmode` to call back into the interpreter and essentially create a
        // continued Python execution environment/stack from there.
        while(tools) {
            // The tools registered are set as bits in `tools` and provide an
            // index into monitoring_callables. This is presumably used by
            // cPython to detect if the slot of a tool type is already in use so
            // that a user can't register more than one tool of a given type at
            // the same time.
            int tool = msb(tools);
            tools ^= (1 << tool);
            // Get the instrument at offset `tool` for the event of interest,
            // this is a callback function, it also might not be present! It
            // is entirely legitimate to have events that have no callback
            // and callbacks that have no event. This is to make it relatively
            // easy to switch events on and off and ensure that monitoring is
            // "lightweight".
            PyObject * instrument = (PyObject *)tstate->interp->monitoring_callables[tool][event];
            if (instrument == NULL){
                continue;
            }

            // Swap the threadstate "event" for the event of interest and
            // increment the tracing tracking field (essentially, inlined
            // PyThreadState_EnterTracing).
            int old_what = tstate->what_event;
            tstate->what_event = event;
            tstate->tracing++;

            // Need to call the callback instrument. Need to know the number of
            // arguments, this is based on whether the `retval` (return value)
            // is NULL (it indicates whether this is a PY_START, or something
            // like a PY_RETURN, which has 3 arguments).
            size_t nargsf = (retval == NULL ? 2 : 3) | PY_VECTORCALL_ARGUMENTS_OFFSET;

            // call the instrumentation, look at the args to the callback
            // functions for sys.monitoring events to find out what the
            // arguments are. e.g.
            // PY_START has `func(code: CodeType, instruction_offset: int)`
            // whereas
            // PY_RETURN has `func(code: CodeType, instruction_offset: int, retval: object)`
            // and
            // CALL, C_RAISE, C_RETURN has `func(code: CodeType, instruction_offset: int, callable: object, arg0 object|MISSING)`
            // i.e. the signature changes based on context. This influences the
            // value of `nargsf` and what is wired into `callback_args`. First two
            // arguments are always code and offset, optional third arg is
            // the return value.
            callback_args[0] = (PyObject*)code;
            callback_args[1] = (PyObject*)offset_obj;
            callback_args[2] = (PyObject*)retval;
            PyObject ** callargs = &callback_args[0];

            // finally, stage the call the the instrument
            result = PyObject_Vectorcall(instrument, callargs, nargsf, NULL);

            // decrement the tracing tracking field and set the event back to
            // the original event (essentially, inlined
            // PyThreadState_LeaveTracing).
            tstate->tracing--;
            tstate->what_event = old_what;

            if (result == NULL){
                // Error occurred in call to instrumentation.
                Py_XDECREF(offset_obj);
                Py_XDECREF(code);
                return -1;
            }
        }
        Py_XDECREF(offset_obj);
        Py_XDECREF(code);
    }
    return 0;
}

/* invoke monitoring for PY_START if it is set */
int static inline invoke_monitoring_PY_START(PyThreadState * tstate, Dispatcher *self) {
    return invoke_monitoring(tstate, PY_MONITORING_EVENT_PY_START, self, NULL);
}

/* invoke monitoring for PY_RETURN if it is set */
int static inline invoke_monitoring_PY_RETURN(PyThreadState * tstate, Dispatcher *self, PyObject * retval) {
    return invoke_monitoring(tstate, PY_MONITORING_EVENT_PY_RETURN, self, retval);
}

/* invoke monitoring for RAISE if it is set */
int static inline invoke_monitoring_RAISE(PyThreadState * tstate, Dispatcher *self, PyObject * exception) {
    return invoke_monitoring(tstate, PY_MONITORING_EVENT_RAISE, self, exception);
}

/* invoke monitoring for PY_UNWIND if it is set */
int static inline invoke_monitoring_PY_UNWIND(PyThreadState * tstate, Dispatcher *self, PyObject * exception) {
    return invoke_monitoring(tstate, PY_MONITORING_EVENT_PY_UNWIND, self, exception);
}

/* A custom, fast, inlinable version of PyCFunction_Call() */
PyObject *
call_cfunc(Dispatcher *self, PyObject *cfunc, PyObject *args, PyObject *kws, PyObject *locals)
{
    PyCFunctionWithKeywords fn = NULL;
    PyThreadState *tstate = NULL;
    PyObject * pyresult = NULL;
    PyObject * pyexception = NULL;

    assert(PyCFunction_Check(cfunc));
    assert(PyCFunction_GET_FLAGS(cfunc) == (METH_VARARGS | METH_KEYWORDS));
    fn = (PyCFunctionWithKeywords) PyCFunction_GET_FUNCTION(cfunc);
    tstate = PyThreadState_GET();
    // issue PY_START if event is set
    if(invoke_monitoring_PY_START(tstate, self) != 0){
        return NULL;
    }
    // make call
    pyresult = fn(PyCFunction_GET_SELF(cfunc), args, kws);
    if (pyresult == NULL) {
        // pyresult == NULL, which means the Numba function raised an exception
        // which is now pending.
        //
        // NOTE: that _ALL_ exceptions trigger the RAISE event, even a
        // StopIteration exception. To get a STOP_ITERATION event, the
        // StopIteration exception must be "implied" i.e. a for loop exhausting
        // a generator, whereas those coming from the executing the binary
        // wrapped in this dispatcher must always be explicit (this is after all
        // a function dispatcher).
        //
        // NOTE: That it is necessary to trigger both a `RAISE` event, as this
        // triggered by an exception being raised, and a `PY_UNWIND` event, as
        // this is the event for  "exiting from a python function during
        // exception unwinding" (see CPython sys.monitoring docs).
        //
        // In the following, if the call to PyErr_GetRaisedException returns
        // NULL, it means that something has cleared the error indicator and
        // this is a most surprising state to occur (shouldn't be possible!).
        //
        // TODO: This makes the exception raising path a little slower as the
        // exception state is suspended and resumed regardless of whether
        // monitoring for such an event is set. In future it might be worth
        // checking the tstate->interp->monitors.tools[event] and only doing the
        // suspend/resume if something is listening for the event.
        pyexception = PyErr_GetRaisedException();
        if (pyexception != NULL) {
            if(invoke_monitoring_RAISE(tstate, self, pyexception) != 0){
                // If the monitoring callback raised, return NULL so that the
                // exception can propagate.
                return NULL;
            }
            if(invoke_monitoring_PY_UNWIND(tstate, self, pyexception) != 0){
                // If the monitoring callback raised, return NULL so that the
                // exception can propagate.
                return NULL;
            }
            // reset the exception
            PyErr_SetRaisedException(pyexception);
        }
        // Exception in Numba call as pyresult == NULL, start to unwind by
        // returning NULL.
        return NULL;
    }
    // issue PY_RETURN if event is set
    if(invoke_monitoring_PY_RETURN(tstate, self, pyresult) != 0){
        return NULL;
    }
    return pyresult;
}