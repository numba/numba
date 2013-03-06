cdef extern from "Python.h":
    ctypedef unsigned int Py_uintptr_t

cdef extern from *:
    ctypedef char *string_t "const char *"

cdef extern from "globalinterning.h":
    ctypedef void *intern_table_t

    intern_table_t intern_create_table() except NULL
    void intern_destroy_table(intern_table_t table)
    string_t intern_key(intern_table_t table, string_t key) except NULL

    int PyIntern_Initialize() except -1
    string_t PyIntern_AddKey(string_t key) except NULL
