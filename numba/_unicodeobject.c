void* get_function_ptr_by_name(const char const* name)
{
    void* result = NULL;
    int i = 0;

    while (PyUnicode_Type.tp_methods[i].ml_name != NULL)
    {
        if (strcmp(PyUnicode_Type.tp_methods[i].ml_name, name) != 0)
        {
            i++;
        }
        else
        {
            result = PyUnicode_Type.tp_methods[i].ml_meth;
            break;
        }
    }

    if (result == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "Cannot find %s function", name);
        abort();  // ToDo: investigate why numba crash with null pointer error
    }

    return result;
}


PyObject* numba_unicode_isalpha(PyObject *self)
{
    PyObject* result;
    PyGILState_STATE gstate;

    static PyObject* (*function_ptr)(PyObject*, PyObject*) = NULL;
    if (function_ptr == NULL)
    {
        function_ptr = get_function_ptr_by_name("isalpha");
        if (function_ptr == NULL) return NULL;
    }

    gstate = PyGILState_Ensure();
    result = (*function_ptr)(self, NULL);
    PyGILState_Release(gstate);

    return result;
}

