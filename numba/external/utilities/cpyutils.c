/* Utilities for use with the CPython C-API */

/* Logic for the raise statement (too complicated for inlining).
   Based on ceval.c:do_raise
*/
static int
do_raise(PyObject *type, PyObject *value, PyObject *tb)
{
    if (type == NULL) {
        /* Reraise */
        PyThreadState *tstate = PyThreadState_GET();
        type = tstate->exc_type == NULL ? Py_None : tstate->exc_type;
        value = tstate->exc_value;
        tb = tstate->exc_traceback;
        Py_XINCREF(type);
        Py_XINCREF(value);
        Py_XINCREF(tb);
    }

    /* We support the following forms of raise:
       raise <class>, <classinstance>
       raise <class>, <argument tuple>
       raise <class>, None
       raise <class>, <argument>
       raise <classinstance>, None
       raise <string>, <object>
       raise <string>, None

       An omitted second argument is the same as None.

       In addition, raise <tuple>, <anything> is the same as
       raising the tuple's first item (and it better have one!);
       this rule is applied recursively.

       Finally, an optional third argument can be supplied, which
       gives the traceback to be substituted (useful when
       re-raising an exception after examining it).  */

    /* First, check the traceback argument, replacing None with
       NULL. */
    if (tb == Py_None) {
        Py_DECREF(tb);
        tb = NULL;
    }
    else if (tb != NULL && !PyTraceBack_Check(tb)) {
        PyErr_SetString(PyExc_TypeError,
                   "raise: arg 3 must be a traceback or None");
        goto raise_error;
    }

    /* Next, replace a missing value with None */
    if (value == NULL) {
        value = Py_None;
        Py_INCREF(value);
    }

    /* Next, repeatedly, replace a tuple exception with its first item */
    while (PyTuple_Check(type) && PyTuple_Size(type) > 0) {
        PyObject *tmp = type;
        type = PyTuple_GET_ITEM(type, 0);
        Py_INCREF(type);
        Py_DECREF(tmp);
    }

    if (PyExceptionClass_Check(type))
        PyErr_NormalizeException(&type, &value, &tb);

    else if (PyExceptionInstance_Check(type)) {
        /* Raising an instance.  The value should be a dummy. */
        if (value != Py_None) {
            PyErr_SetString(PyExc_TypeError,
              "instance exception may not have a separate value");
            goto raise_error;
        }
        else {
            /* Normalize to raise <class>, <instance> */
            Py_DECREF(value);
            value = type;
            type = PyExceptionInstance_Class(type);
            Py_INCREF(type);
        }
    }
    else {
        /* Not something you can raise.  You get an exception
           anyway, just not what you specified :-) */
        PyErr_Format(PyExc_TypeError,
                     "exceptions must be old-style classes or "
                     "derived from BaseException, not %s",
                     type->ob_type->tp_name);
        goto raise_error;
    }

    assert(PyExceptionClass_Check(type));

    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(tb);
    PyErr_Restore(type, value, tb);
 raise_error:
    return -1;
}

static int
export_cpyutils(PyObject *module)
{
    EXPORT_FUNCTION(do_raise, module, error)

    return 0;
error:
    return -1;
}