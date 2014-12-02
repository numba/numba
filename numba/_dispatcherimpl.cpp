#include "typeconv/typeconv.hpp"
#include <cassert>
#include <vector>

typedef std::vector<Type> TypeTable;
typedef std::vector<void*> Functions;

struct _opaque_dispatcher {};

class Dispatcher: public _opaque_dispatcher {
public:
    Dispatcher(TypeManager *tm, int argct): argct(argct), tm(tm) { }

    void addDefinition(Type args[], void *callable) {
        overloads.reserve(argct + overloads.size());
        for (int i=0; i<argct; ++i) {
            overloads.push_back(args[i]);
        }
        functions.push_back(callable);
    }

    void* resolve(Type sig[], int &matches, bool allow_unsafe) {
        const int ovct = functions.size();
        int selected;
        matches = 0;
        if (0 == ovct) {
            return NULL;
        }
        if (overloads.size() > 0) {
            matches = tm->selectOverload(sig, &overloads[0], selected, argct,
                                         ovct, allow_unsafe);
        } else if (argct == 0){
            matches = 1;
            selected = 0;
        }
        if (matches == 1){
            return functions[selected];
        }
        return NULL;
    }

    int count() const { return functions.size(); }

    const int argct;
private:
    TypeManager *tm;
    TypeTable overloads;
    Functions functions;
};


#include "_dispatcher.h"

dispatcher_t *
dispatcher_new(void *tm, int argct){
    return new Dispatcher(static_cast<TypeManager*>(tm), argct);
}

void
dispatcher_del(dispatcher_t *obj) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    delete disp;
}

void
dispatcher_add_defn(dispatcher_t *obj, int tys[], void* callable) {
    assert(sizeof(int) == sizeof(Type) &&
            "Type should be representable by an int");

    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    Type *args = reinterpret_cast<Type*>(tys);
    disp->addDefinition(args, callable);
}

void*
dispatcher_resolve(dispatcher_t *obj, int sig[], int *count, int allow_unsafe) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    Type *args = reinterpret_cast<Type*>(sig);
    void *callable = disp->resolve(args, *count, (bool) allow_unsafe );
    return callable;
}

int
dispatcher_count(dispatcher_t *obj) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    return disp->count();
}

// Structured type cache implementation

#include <map>
#include <set>
#include "_pymodule.h"

struct record_field {
    /* The name of the field - may be UCS1, UCS2 or UCS4 as indicated by kind */
    void* name;
    /* Kind of the name string - values as PyUnicode_KIND(o) */
    int kind;
    /* Length of name in bytes */
    Py_ssize_t length;
    /* Type num of the field's data type */
    int type_num;
    /* Offset of the field into the record */
    int offset;

    record_field(void* name, int kind, Py_ssize_t length, int type_num,
                 int offset)
        : name(name), kind(kind), length(length), type_num(type_num),
          offset(offset) { }

    record_field(const record_field& other): kind(other.kind),
                                             length(other.length),
                                             type_num(other.type_num),
                                             offset(other.offset) {
        name = malloc(length);
        memcpy(name, other.name, length);
    }

    ~record_field() {
        free(name);
    }

    bool operator<(const record_field& other) const {
        if (offset < other.offset)
            return true;
        else if (offset > other.offset)
            return false;

        if (type_num < other.type_num)
            return true;
        else if (type_num > other.type_num)
            return false;

        if (length < other.length)
            return true;
        else if (length > other.length)
            return false;

        if (kind < other.kind)
            return true;
        else if (kind > other.kind)
            return false;

        return (memcmp(name, other.name, length) < 0);
    }
};

typedef std::set<record_field> Record;

// Convert a PyArray_Descr to a Record. Return of an empty record indicates
// a failure in creating the record.
static
Record descr_to_record(PyArray_Descr* descr) {
    PyObject* fields = descr->fields;
    PyObject* keys = PyDict_Keys(fields);

    Record record;

    for (int i = 0; i < PyList_Size(keys); ++i) {
        PyObject* key = PyList_GET_ITEM(keys, i);
        // Ensure all keys are in the canonical state
        if (PyUnicode_READY(key) == -1) {
            Py_DECREF(keys);
            return Record();
        }

        PyObject* value = PyDict_GetItem(fields, key);

        void* data = PyUnicode_DATA(key);
        Py_ssize_t length = PyUnicode_GET_LENGTH(key);
        int kind = PyUnicode_KIND(key);
        Py_ssize_t datalen = length * kind;
        void* name = malloc(datalen);
        memcpy(name, data, datalen);

        int type_num = ((PyArray_Descr*)PyTuple_GET_ITEM(value, 0))->type_num;
        // We do not handle records containing fields that don't have one
        // of the basic typecodes
        if (dtype_num_to_typecode(type_num) == -1) {
            Py_DECREF(keys);
            return Record();
        }

        int offset = PyLong_AsLong(PyTuple_GET_ITEM(value, 1));
        record_field rf(name, kind, datalen, type_num, offset);
        record.insert(rf);
    }

    Py_DECREF(keys);
    return record;
}

// NDArray type cache

struct ndarray_type {
    int ndim;
    int layout;
    Record record;
    ndarray_type(int ndim, int layout, Record record)
        : ndim(ndim), layout(layout), record(record) { }

    bool operator<(const ndarray_type &other) const {
        if (ndim < other.ndim)
            return true;
        else if (ndim > other.ndim)
            return false;

        if (layout < other.layout)
            return true;
        else if (layout > other.layout)
            return false;

        if (record < other.record)
            return true;
        else
            return false;
    }
};

typedef std::map<ndarray_type, int> NDArrayTypeMap;
static NDArrayTypeMap ndarray_typemap;

int
dispatcher_get_ndarray_typecode(int ndim, int layout, PyArray_Descr* descr) {
    Record r = descr_to_record(descr);
    if (r.empty())
        return -1;

    ndarray_type k(ndim, layout, r);
    NDArrayTypeMap::iterator i = ndarray_typemap.find(k);
    if (i == ndarray_typemap.end()) {
        return -1;
    }

    return i->second;
}

void
dispatcher_insert_ndarray_typecode(int ndim, int layout,
                                   PyArray_Descr* descr, int typecode) {
    Record r = descr_to_record(descr);
    if (r.empty())
        return;

    ndarray_type k(ndim, layout, r);
    ndarray_typemap[k] = typecode;
}

typedef std::map<Record, int> ArrayScalarTypeMap;
static ArrayScalarTypeMap arrayscalar_typemap;

int dispatcher_get_arrayscalar_typecode(PyArray_Descr* descr) {
    Record r = descr_to_record(descr);
    if (r.empty())
        return -1;

    ArrayScalarTypeMap::iterator i = arrayscalar_typemap.find(r);
    if (i == arrayscalar_typemap.end()) {
        return -1;
    }

    return i->second;
}

void dispatcher_insert_arrayscalar_typecode(PyArray_Descr *descr, int typecode) {
    Record r = descr_to_record(descr);

    if (!r.empty())
        arrayscalar_typemap[r] = typecode;
}
