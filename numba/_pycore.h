#include <stdbool.h>
#include <stddef.h>

#ifndef Py_INTERNAL_H
#define Py_INTERNAL_H
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/3c137dc613c860f605d3520d7fd722cd8ed79da6/Include/internal/pycore_pyerrors.h
 */

PyAPI_FUNC(void) _PyErr_Fetch(
    PyThreadState *tstate,
    PyObject **type,
    PyObject **value,
    PyObject **traceback);

PyAPI_FUNC(void) _PyErr_Restore(
    PyThreadState *tstate,
    PyObject *type,
    PyObject *value,
    PyObject *traceback);

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/3c137dc613c860f605d3520d7fd722cd8ed79da6/Include/internal/pycore_code.h
 */

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

/*
 * Code originally from:
 * https://github.com/python/cpython/blob/e72f469e857e2e853dd067742e4c8c5f7bb8fb16/Include/internal/pycore_frame.h
 */

/* See Objects/frame_layout.md for an explanation of the frame stack
 * including explanation of the PyFrameObject and _PyInterpreterFrame
 * structs. */

#define _PyInterpreterFrame_LASTI(IF) \
    ((int)((IF)->prev_instr - _PyCode_CODE((IF)->f_code)))

struct _frame {
    PyObject_HEAD
    PyFrameObject *f_back;      /* previous frame, or NULL */
    struct _PyInterpreterFrame *f_frame; /* points to the frame data */
    PyObject *f_trace;          /* Trace function */
    int f_lineno;               /* Current line number. Only valid if non-zero */
    char f_trace_lines;         /* Emit per-line trace events? */
    char f_trace_opcodes;       /* Emit per-opcode trace events? */
    char f_fast_as_locals;      /* Have the fast locals of this frame been converted to a dict? */
    /* The frame data, if this frame object owns the frame */
    PyObject *_f_frame_data[1];
};

extern PyFrameObject* _PyFrame_New_NoTrack(PyCodeObject *code);

/* other API */

typedef enum _framestate {
    FRAME_CREATED = -2,
    FRAME_SUSPENDED = -1,
    FRAME_EXECUTING = 0,
    FRAME_COMPLETED = 1,
    FRAME_CLEARED = 4
} PyFrameState;

enum _frameowner {
    FRAME_OWNED_BY_THREAD = 0,
    FRAME_OWNED_BY_GENERATOR = 1,
    FRAME_OWNED_BY_FRAME_OBJECT = 2
};

typedef struct _PyInterpreterFrame {
    /* "Specials" section */
    PyFunctionObject *f_func; /* Strong reference */
    PyObject *f_globals; /* Borrowed reference */
    PyObject *f_builtins; /* Borrowed reference */
    PyObject *f_locals; /* Strong reference, may be NULL */
    PyCodeObject *f_code; /* Strong reference */
    PyFrameObject *frame_obj; /* Strong reference, may be NULL */
    /* Linkage section */
    struct _PyInterpreterFrame *previous;
    // NOTE: This is not necessarily the last instruction started in the given
    // frame. Rather, it is the code unit *prior to* the *next* instruction. For
    // example, it may be an inline CACHE entry, an instruction we just jumped
    // over, or (in the case of a newly-created frame) a totally invalid value:
    _Py_CODEUNIT *prev_instr;
    int stacktop;     /* Offset of TOS from localsplus  */
    bool is_entry;  // Whether this is the "root" frame for the current _PyCFrame.
    char owner;
    /* Locals and stack */
    PyObject *localsplus[1];
} _PyInterpreterFrame;

/* Determine whether a frame is incomplete.
 * A frame is incomplete if it is part way through
 * creating cell objects or a generator or coroutine.
 *
 * Frames on the frame stack are incomplete until the
 * first RESUME instruction.
 * Frames owned by a generator are always complete.
 */
static inline bool
_PyFrame_IsIncomplete(_PyInterpreterFrame *frame)
{
    return frame->owner != FRAME_OWNED_BY_GENERATOR &&
    frame->prev_instr < _PyCode_CODE(frame->f_code) + frame->f_code->_co_firsttraceable;
}

/* For use by _PyFrame_GetFrameObject
  Do not call directly. */
PyFrameObject *
_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame *frame);

/* Gets the PyFrameObject for this frame, lazily
 * creating it if necessary.
 * Returns a borrowed referennce */
static inline PyFrameObject *
_PyFrame_GetFrameObject(_PyInterpreterFrame *frame)
{

    assert(!_PyFrame_IsIncomplete(frame));
    PyFrameObject *res =  frame->frame_obj;
    if (res != NULL) {
        return res;
    }
    return _PyFrame_MakeAndSetFrameObject(frame);
}

#ifdef __cplusplus
}
#endif
#endif /* PYCORE_INTERNAL_H */