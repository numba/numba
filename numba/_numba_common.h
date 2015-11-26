#ifndef NUMBA_COMMON_H_
#define NUMBA_COMMON_H_

/* __has_attribute() is a clang / gcc-5 macro */
#ifndef __has_attribute
#   define __has_attribute(x) 0
#endif

/* This attribute marks symbols that can be shared accross C objects
 * but are not exposed outside of a shared library or executable.
 * Note this is default behaviour for global symbols under Windows.
 */
#if (__has_attribute(visibility) || \
     (defined(__GNUC__) && __GNUC__ >= 4))
#define VISIBILITY_HIDDEN __attribute__ ((visibility("hidden")))
#else
#define VISIBILITY_HIDDEN
#endif

#endif /* NUMBA_COMMON_H_ */
