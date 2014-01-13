#include "Python.h"
#include "numpy/arrayobject.h"

/*
    Minivect utilities

    get_arrays_ordering() returns the flags indicating which minivect specialization
    should be used.
*/

/* Data layout constants. Used to determine the specialization. */
#define ARRAY_C_ORDER 0x1

#define ARRAYS_ARE_CONTIG 0x2
#define ARRAYS_ARE_INNER_CONTIG 0x4
#define ARRAYS_ARE_MIXED_CONTIG 0x10
#define ARRAYS_ARE_STRIDED 0x20
#define ARRAYS_ARE_MIXED_STRIDED 0x40


#define absval(val) (val < 0 ? -val : val)

/*
    Figure out the best memory access order for a given array, ignore broadcasting.
*/
static char
get_best_order(PyArrayObject *array, int ndim)
{
    int i, j;

    npy_intp *shape = PyArray_DIMS(array);

    npy_intp c_stride = 0;
    npy_intp f_stride = 0;

    if (ndim == 1)
        return 'A'; /* short-circuit */

    for (i = ndim - 1; i >= 0; i--) {
        if (shape[i] != 1) {
            c_stride = PyArray_STRIDE(array, i);
            break;
        }
    }

    for (j = 0; j < ndim; j++) {
        if (shape[j] != 1) {
            f_stride = PyArray_STRIDE(array, j);
            break;
        }
    }

    if (i == j) {
        if (i > 0)
            return 'c';
        else
            return 'f';
    } else if (absval(c_stride) <= absval(f_stride)) {
        return 'C';
    } else {
        return 'F';
    }
}

/*
    Get the overall data order for a list of NumPy arrays for
    element-wise traversal.
*/
static PyObject *
get_arrays_ordering(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    int result_flags;

    int all_c_contig = 1;
    int all_f_contig = 1;
    int seen_c_contig = 0;
    int seen_f_contig = 0;
    int seen_c_ish = 0;
    int seen_f_ish = 0;

    int i;

    PyObject *arrays;
    int n_arrays;
    int broadcasting = 0;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &arrays)) {
        return NULL;
    }

    n_arrays = PyList_GET_SIZE(arrays);
    if (!n_arrays) {
        PyErr_SetString(PyExc_ValueError, "Expected non-empty list of arrays");
        return NULL;
    }

    /* Count the data orders */
    for (i = 0; i < n_arrays; i++) {
        char order;

        PyArrayObject *array = (PyArrayObject *) PyList_GET_ITEM(arrays, i);
        int ndim = PyArray_NDIM(array);
        int c_contig = PyArray_ISCONTIGUOUS(array);
        int f_contig = PyArray_ISFORTRAN(array);

        if (!ndim)
            continue;

        if (c_contig) {
            order = 'C';
        } else if (f_contig) {
            order = 'F';
        } else {
            order = get_best_order(array, ndim);

            if (order == 'c' || order == 'f') {
                broadcasting++;
                order = toupper(order);
            }
        }

        if (order == 'C') {
            all_f_contig = 0;
            all_c_contig &= c_contig;
            seen_c_contig += c_contig;
            seen_c_ish++;
        } else {
            all_c_contig = 0;
            all_f_contig &= f_contig;
            seen_f_contig += f_contig;
            seen_f_ish++;
        }
    }

    if (all_c_contig || all_f_contig) {
        result_flags = ARRAYS_ARE_CONTIG | all_c_contig;
    } else if (broadcasting == n_arrays) {
        result_flags = ARRAYS_ARE_STRIDED | ARRAY_C_ORDER;
    } else if (seen_c_contig + seen_f_contig == n_arrays - broadcasting) {
        result_flags = ARRAYS_ARE_MIXED_CONTIG | (seen_c_ish > seen_f_ish);
    } else if (seen_c_ish && seen_f_ish) {
        result_flags = ARRAYS_ARE_MIXED_STRIDED | (seen_c_ish > seen_f_ish);
    } else {
        /*
           Check whether the operands are strided or inner contiguous.
           We check whether the stride in the first or last (F/C) dimension equals
           the itemsize, and we verify that no operand is broadcasting in the
           first or last (F/C) dimension (that they all have the same extent).
        */
        PyArrayObject *array = (PyArrayObject *) PyList_GET_ITEM(arrays, 0);
        npy_intp extent;

        if (seen_c_ish)
            extent = PyArray_DIM(array, PyArray_NDIM(array) - 1);
        else
            extent = PyArray_DIM(array, 0);

        /* Assume inner contiguous */
        result_flags = ARRAYS_ARE_INNER_CONTIG | !!seen_c_ish;
        for (i = 0; i < n_arrays; i++) {
            int dim = 0;
            array = (PyArrayObject *) PyList_GET_ITEM(arrays, i);
            if (seen_c_ish)
                dim = PyArray_NDIM(array) - 1;

            if (dim < 0)
                continue;

            if (PyArray_STRIDE(array, dim) != PyArray_ITEMSIZE(array) ||
                    PyArray_DIM(array, dim) != extent) {
                result_flags = ARRAYS_ARE_STRIDED | !!seen_c_ish;
                break;
            }
        }
    }

    return PyLong_FromLong(result_flags);
}

/*
    The below function adds the data layout constants to a Python module.
    Call from the module init function.
*/
static int
add_array_order_constants(PyObject *module)
{
#define __err_if_neg(expr) if (expr < 0) return -1;
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAY_C_ORDER", ARRAY_C_ORDER));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_CONTIG", ARRAYS_ARE_CONTIG));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_INNER_CONTIG", ARRAYS_ARE_INNER_CONTIG));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_MIXED_CONTIG", ARRAYS_ARE_MIXED_CONTIG));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_STRIDED", ARRAYS_ARE_STRIDED));
    __err_if_neg(PyModule_AddIntConstant(module, "ARRAYS_ARE_MIXED_STRIDED", ARRAYS_ARE_MIXED_STRIDED));
#undef __err_if_neg
    return 0;
}

/* Use the code below to add get_arrays_ordering as a function in an extension module */
/*
static PyMethodDef ext_methods[] = {
#if  PY_MAJOR_VERSION >= 3
    {"get_arrays_ordering", (PyCFunction) get_arrays_ordering, METH_VARARGS, NULL},
#else
    {"get_arrays_ordering", get_arrays_ordering, METH_VARARGS, NULL},
#endif
    { NULL }
};
*/
