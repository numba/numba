#ifndef NUMBA_TYPEOF_H_
#define NUMBA_TYPEOF_H_


extern PyObject *typeof_init(PyObject *self, PyObject *args);
extern int typeof_typecode(PyObject *dispatcher, PyObject *val);
extern PyObject *typeof_compute_fingerprint(PyObject *val);


#endif  /* NUMBA_TYPEOF_H_ */
