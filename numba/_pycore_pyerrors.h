/*
 * Code originally from:
 * https://github.com/python/cpython/blob/3c137dc613c860f605d3520d7fd722cd8ed79da6/Include/internal/pycore_pyerrors.h
 */

#ifndef Py_INTERNAL_PYERRORS_H
#define Py_INTERNAL_PYERRORS_H
#ifdef __cplusplus
extern "C" {
#endif

PyAPI_FUNC(void) _PyErr_Fetch(
    PyThreadState *tstate,
    PyObject **type,
    PyObject **value,
    PyObject **traceback);

PyAPI_FUNC(void) _PyErr_Restore(
    PyThreadState *tstate,
    PyObject *type,
    PyObject *value,
    PyObject *traceback);
#ifdef __cplusplus

}
#endif
#endif /* !Py_INTERNAL_PYERRORS_H */
