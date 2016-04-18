#include "_pymodule.h"

typedef struct {
    PyObject_HEAD
    PyObject *meminfo;
    PyObject *meminfoptr;
    PyObject *dataptr;
} BoxObject;


static PyMemberDef box_members[] = {
    {"_meminfo",    T_OBJECT, offsetof(BoxObject, meminfo), READONLY},
    {"_meminfoptr", T_OBJECT, offsetof(BoxObject, meminfoptr), READONLY},
    {"_dataptr",    T_OBJECT, offsetof(BoxObject, dataptr), READONLY},
    {NULL}  /* Sentinel */
};

static PyObject *MemInfoClass = NULL;

static
int Box_init(BoxObject *self, PyObject *args, PyObject *kwds) {
    static char *keywords[] = {"meminfoptr", "dataptr", NULL};
    PyObject *meminfo;
    PyObject *meminfoptr;
    PyObject *dataptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", keywords,
                                     &meminfoptr, &dataptr))
    {
        return -1;
    }

    /* MemInfo(meminfoptr) */
    meminfo = PyObject_CallFunctionObjArgs(MemInfoClass, meminfoptr, NULL);
    if (!meminfo) return -1;

    /* Set attributes */
    Py_INCREF(meminfoptr);
    Py_INCREF(dataptr);
    self->meminfo = meminfo;
    self->meminfoptr = meminfoptr;
    self->dataptr = dataptr;
    return 0;
}


static void
box_dealloc(BoxObject *box)
{
    Py_DECREF(box->meminfo);
    Py_DECREF(box->meminfoptr);
    Py_DECREF(box->dataptr);
    Py_TYPE(box)->tp_free((PyObject *) box);
}


static const char Box_doc[] = "A box for numba created jit-class instance";


static PyTypeObject BoxType = {
#if (PY_MAJOR_VERSION < 3)
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#else
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "_box.Box",                /*tp_name*/
    sizeof(BoxObject),         /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)box_dealloc,   /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    Box_doc,                   /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    box_members,               /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Box_init,        /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};


static
PyObject * import_meminfo_class() {
    PyObject *nrtmod;
    PyObject *meminfocls;
    /* from numba.runtime.nrt import MemInfo */
    nrtmod = PyImport_ImportModule("numba.runtime.nrt");
    if (!nrtmod) return NULL;
    meminfocls = PyObject_GetAttrString(nrtmod, "MemInfo");
    if (!meminfocls) {
        Py_DECREF(nrtmod);
        return NULL;
    }
    return meminfocls;
}

MOD_INIT(_box) {
    PyObject *m;

    MOD_DEF(m, "_box", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    /* init BoxType */
    BoxType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&BoxType))
        return MOD_ERROR_VAL;

    /* import and cache numba.runtime.nrt.MemInfo
       and keep it in the module */
    MemInfoClass = import_meminfo_class();
    if (!MemInfoClass) return MOD_ERROR_VAL;
    Py_INCREF(MemInfoClass);
    PyModule_AddObject(m, "_MemInfo", MemInfoClass);

    /* bind BoxType */
    Py_INCREF(&BoxType);
    PyModule_AddObject(m, "Box", (PyObject *) (&BoxType));

    return MOD_SUCCESS_VAL(m);
}
