#include "../_pymodule.h"
#include "../capsulethunk.h"
#include "typeconv.hpp"

extern "C" {


static PyObject*
new_type_manager(PyObject* self, PyObject* args);

static void
del_type_manager(PyObject *);

static PyObject*
select_overload(PyObject* self, PyObject* args);

static PyObject*
check_compatible(PyObject* self, PyObject* args);

static PyObject*
set_compatible(PyObject* self, PyObject* args);

static PyObject*
get_pointer(PyObject* self, PyObject* args);


static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(new_type_manager),
    declmethod(select_overload),
    declmethod(check_compatible),
    declmethod(set_compatible),
    declmethod(get_pointer),
    { NULL },
#undef declmethod
};


MOD_INIT(_typeconv) {
    PyObject *m;
    MOD_DEF(m, "_typeconv", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}

} // end extern C

///////////////////////////////////////////////////////////////////////////////

const char PY_CAPSULE_TM_NAME[] = "*tm";
#define BAD_TM_ARGUMENT PyErr_SetString(PyExc_TypeError,                    \
                                        "1st argument not TypeManager")

static
TypeManager* unwrap_TypeManager(PyObject *tm) {
    void* p = PyCapsule_GetPointer(tm, PY_CAPSULE_TM_NAME);
    return reinterpret_cast<TypeManager*>(p);
}

PyObject*
new_type_manager(PyObject* self, PyObject* args)
{
    TypeManager* tm = new TypeManager();
    return PyCapsule_New(tm, PY_CAPSULE_TM_NAME, &del_type_manager);
}

void
del_type_manager(PyObject *tm)
{
    delete unwrap_TypeManager(tm);
}

PyObject*
select_overload(PyObject* self, PyObject* args)
{
    PyObject *tmcap, *sigtup, *ovsigstup;
    int allow_unsafe;

    if (!PyArg_ParseTuple(args, "OOOi", &tmcap, &sigtup, &ovsigstup,
                          &allow_unsafe)) {
        return NULL;
    }

    TypeManager *tm = unwrap_TypeManager(tmcap);
    if (!tm) {
        BAD_TM_ARGUMENT;
    }

    Py_ssize_t sigsz = PySequence_Size(sigtup);
    Py_ssize_t ovsz = PySequence_Size(ovsigstup);

    Type *sig = new Type[sigsz];
    Type *ovsigs = new Type[ovsz * sigsz];

    for (int i = 0; i < sigsz; ++i) {
        sig[i] = Type(PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(sigtup,
                                                                  i), NULL));
    }

    for (int i = 0; i < ovsz; ++i) {
        PyObject *cursig = PySequence_Fast_GET_ITEM(ovsigstup, i);
        for (int j = 0; j < sigsz; ++j) {
            long tid = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(cursig,
                                                                   j), NULL);
            ovsigs[i * sigsz + j] = Type(tid);
        }
    }

    int selected = -42;
    int matches = tm->selectOverload(sig, ovsigs, selected, sigsz, ovsz,
                                     (bool) allow_unsafe);

    delete [] sig;
    delete [] ovsigs;

    if (matches > 1) {
        PyErr_SetString(PyExc_TypeError, "Ambigous overloading");
        return NULL;
    } else if (matches == 0) {
        PyErr_SetString(PyExc_TypeError, "No compatible overload");
        return NULL;
    }

    return PyLong_FromLong(selected);
}

PyObject*
check_compatible(PyObject* self, PyObject* args)
{
    PyObject *tmcap;
    int from, to;
    if (!PyArg_ParseTuple(args, "Oii", &tmcap, &from, &to)) {
        return NULL;
    }

    TypeManager *tm = unwrap_TypeManager(tmcap);
    if(!tm) {
        BAD_TM_ARGUMENT;
        return NULL;
    }

    switch(tm->isCompatible(Type(from), Type(to))){
    case TCC_EXACT:
        return PyString_FromString("exact");
    case TCC_PROMOTE:
        return PyString_FromString("promote");
    case TCC_CONVERT_SAFE:
        return PyString_FromString("safe");
        case TCC_CONVERT_UNSAFE:
        return PyString_FromString("unsafe");
    default:
        Py_RETURN_NONE;
    }
}

PyObject*
set_compatible(PyObject* self, PyObject* args)
{
    PyObject *tmcap;
    int from, to, by;
    if (!PyArg_ParseTuple(args, "Oiii", &tmcap, &from, &to, &by)) {
        return NULL;
    }

    TypeManager *tm = unwrap_TypeManager(tmcap);
    if (!tm) {
        BAD_TM_ARGUMENT;
        return NULL;
    }
    TypeCompatibleCode tcc;
    switch (by) {
    case 'p': // promote
        tcc = TCC_PROMOTE;
        break;
    case 's': // safe convert
        tcc = TCC_CONVERT_SAFE;
        break;
    case 'u': // unsafe convert
        tcc = TCC_CONVERT_UNSAFE;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Unknown TCC");
        return NULL;
    }

    tm->addCompatibility(Type(from), Type(to), tcc);
    Py_RETURN_NONE;
}


PyObject*
get_pointer(PyObject* self, PyObject* args)
{
    PyObject *tmcap;
    if (!PyArg_ParseTuple(args, "O", &tmcap)) {
        return NULL;
    }
    return PyLong_FromVoidPtr(unwrap_TypeManager(tmcap));
}


