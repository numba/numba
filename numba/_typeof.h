#ifndef NUMBA_TYPEOF_H_
#define NUMBA_TYPEOF_H_

#ifdef __cplusplus
    extern "C" {
#endif

extern PyObject *typeof_init(PyObject *self, PyObject *args);
extern int typeof_typecode(PyObject *dispatcher, PyObject *val);
extern PyObject *typeof_compute_fingerprint(PyObject *val);

// Define the C-struct representation of OmittedArg
typedef struct {
    PyObject_HEAD
    PyObject *value;  // Direct pointer to the wrapped default value
} OmittedArgObject;

extern PyTypeObject OmittedArgType;

#ifdef __cplusplus
    }
#endif

#endif  /* NUMBA_TYPEOF_H_ */
