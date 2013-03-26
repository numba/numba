def global_intern(bytes key):
    return PyIntern_AddKey(key)

def global_intern_initialize():
    PyIntern_Initialize()

cdef class InternTable(object):
    "Wrap intern tables (intern_table_t)"

    cdef intern_table_t _table
    cdef intern_table_t *table

    def __init__(self):
        self.table = intern_create_table(&self._table)

    def __dealloc__(self):
        intern_destroy_table(self.table)

    def intern(self, bytes key):
        return intern_key(self.table, key)
