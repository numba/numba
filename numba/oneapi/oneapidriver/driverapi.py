from cffi import FFI
import os
import sys


ffibuilder = FFI()

oneapiGlueHome = os.path.dirname(os.path.realpath(__file__))

BAD_ENV_PATH_ERRMSG = """
NUMBA_ONEAPI_GLUE_HOME is set to '{0}' which is not a valid path to a
dynamic link library for your system.
"""


def _raise_bad_env_path(path, extra=None):
    error_message = BAD_ENV_PATH_ERRMSG.format(path)
    if extra is not None:
        error_message += extra
    raise ValueError(error_message)

#oneapiGlueHome = os.environ.get('NUMBA_ONEAPI_GLUE_HOME', None)

# if oneapiGlueHome is None:
#    raise ValueError("FATAL: Set the NUMBA_ONEAPI_GLUE_HOME for "
#                     "numba_oneapi_glue.h and libnumbaoneapiglue.so")


if oneapiGlueHome is not None:
    try:
        oneapi_glue_home = os.path.abspath(oneapiGlueHome)
    except ValueError:
        _raise_bad_env_path(oneapiGlueHome)

    if not os.path.isfile(oneapiGlueHome + "/libnumbaoneapiglue.a"):
        _raise_bad_env_path(oneapiGlueHome + "/libnumbaoneapiglue.a")

glue_h = ''.join(list(filter(lambda x: len(x) > 0 and x[0] != "#", open(oneapiGlueHome + '/numba_oneapi_glue.h', 'r').readlines())))

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef(glue_h)

ffi_lib_name = "numba.oneapi.oneapidriver._numba_oneapi_pybindings"

ffibuilder.set_source(
    ffi_lib_name,
    """
         #include "numba_oneapi_glue.h"   // the C header of the library
    """,
    libraries=["numbaoneapiglue", "OpenCL"],
    include_dirs=[oneapiGlueHome],
    library_dirs=[oneapiGlueHome]
)   # library name, for the linker


if __name__ == "__main__":
    # ffibuilder.emit_c_code("pybindings.c")
    ffibuilder.compile(verbose=True)
