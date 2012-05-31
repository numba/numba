#ifndef Py_CUSTOMSLOTS_H
#define Py_CUSTOMSLOTS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <structmember.h>
#include <stdint.h>
/* Some stdint.h implementations:
Portable: http://www.azillionmonkeys.com/qed/pstdint.h
MSVC: http://msinttypes.googlecode.com/svn/trunk/stdint.h
*/

#if defined(__GNUC__) && (__GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ > 95))
  #define PY_CUSTOMSLOTS_LIKELY(x)   __builtin_expect(!!(x), 1)
  #define PY_CUSTOMSLOTS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define PY_CUSTOMSLOTS_LIKELY(x)   (x)
  #define PY_CUSTOMSLOTS_UNLIKELY(x)   (x)
#endif

/* inline attribute */
#if defined(__GNUC__)
  #define PY_CUSTOMSLOTS_INLINE __inline__
#elif defined(_MSC_VER)
  #define PY_CUSTOMSLOTS_INLINE __inline
#elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
  #define PY_CUSTOMSLOTS_INLINE inline
#else
  #define PY_CUSTOMSLOTS_INLINE
#endif


#define PyExtensibleType_TPFLAGS_IS_EXTENSIBLE (1L<<22)

typedef struct {
  uintptr_t id;
  void *data;
} PyCustomSlot;

typedef struct {
  PyHeapTypeObject etp_heaptype;
  Py_ssize_t etp_custom_slot_count; /* length of tpe_custom_slots array */
  PyCustomSlot *etp_custom_slot_table;
} PyHeapExtensibleTypeObject;

#define PyCustomSlots_Check(obj) \
  (((obj)->ob_type->tp_flags & PyExtensibleType_TPFLAGS_IS_EXTENSIBLE) == \
   PyExtensibleType_TPFLAGS_IS_EXTENSIBLE)

#define PyCustomSlots_Count(obj) \
  (((PyHeapExtensibleTypeObject*)(obj)->ob_type)->etp_custom_slot_count)

#define PyCustomSlots_Table(obj) \
  (((PyHeapExtensibleTypeObject*)(obj)->ob_type)->etp_custom_slot_table)

static PY_CUSTOMSLOTS_INLINE PyCustomSlot *
PyCustomSlots_Find(PyObject *obj,
                   uintptr_t id,
                   Py_ssize_t expected_pos) {
  PyCustomSlot *entries;
  Py_ssize_t i;
  /* We unroll and make hitting the first slot likely(); this saved
     about 2 cycles on the test system with gcc 4.6.3, -O2 */
  if (PY_CUSTOMSLOTS_LIKELY(PyCustomSlots_Check(obj))) {
    entries = PyCustomSlots_Table(obj);
    if (PY_CUSTOMSLOTS_LIKELY(PyCustomSlots_Count(obj) > expected_pos &&
                              entries[expected_pos].id == id)) {
      return &entries[expected_pos];
    } else {
      for (i = 0; i != PyCustomSlots_Count(obj); ++i) {
        if (entries[i].id == id) {
          return &entries[i];
        }
      }
    }
  }
  return 0;
}


#ifdef __cplusplus
}
#endif
#endif /* !Py_CUSTOMSLOTS_H */
