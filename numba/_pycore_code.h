/*
 * Code originally from:
 * https://github.com/python/cpython/blob/3c137dc613c860f605d3520d7fd722cd8ed79da6/Include/internal/pycore_code.h
 */

#ifndef Py_INTERNAL_CODE_H
#define Py_INTERNAL_CODE_H
#ifdef __cplusplus
extern "C" {
#endif

/* Line array cache for tracing */

extern int _PyCode_CreateLineArray(PyCodeObject *co);

static inline int
_PyCode_InitLineArray(PyCodeObject *co)
{
    if (co->_co_linearray) {
        return 0;
    }
    return _PyCode_CreateLineArray(co);
}

static inline int
_PyCode_LineNumberFromArray(PyCodeObject *co, int index)
{
    assert(co->_co_linearray != NULL);
    assert(index >= 0);
    assert(index < Py_SIZE(co));
    if (co->_co_linearray_entry_size == 2) {
        return ((int16_t *)co->_co_linearray)[index];
    }
    else {
        assert(co->_co_linearray_entry_size == 4);
        return ((int32_t *)co->_co_linearray)[index];
    }
}

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_CODE_H */
