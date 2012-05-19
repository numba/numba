#ifndef Py_EXTENSIBLETYPE_H
#define Py_EXTENSIBLETYPE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>

typedef struct {
    unsigned long tpe_extension_id;
    void *tpe_data;
} PyExtensibleTypeObjectEntry;

typedef struct {
  PyTypeObject tpe_base;
  Py_ssize_t tpe_count; /* length of tpe_entries array */
  PyExtensibleTypeObjectEntry tpe_entries[0]; /* variable size array */  
} PyExtensibleTypeObject;

PyTypeObject *PyExtensibleType_GetMetaClass(void);




#ifdef __cplusplus
}
#endif
#endif /* !Py_EXTENSIBLETYPE_H */
