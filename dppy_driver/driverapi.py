from cffi import FFI
import os
import sys


ffibuilder = FFI()

libDPGlueHome = os.path.dirname(os.path.realpath(__file__))

BAD_ENV_PATH_ERRMSG = """
DP_GLUE_HOME is set to '{0}' which is not a valid path to a
dynamic link library for your system.
"""


def _raise_bad_env_path(path, extra=None):
    error_message = BAD_ENV_PATH_ERRMSG.format(path)
    if extra is not None:
        error_message += extra
    raise ValueError(error_message)

#libDPGlueHome = os.environ.get('DP_GLUE_HOME', None)

# if libDPGlueHome is None:
#    raise ValueError("FATAL: Set the DP_GLUE_HOME for "
#                     "dp_glue.h and libDPGlueHome.so")


if libDPGlueHome is not None:
    try:
        os.path.abspath(libDPGlueHome)
    except ValueError:
        _raise_bad_env_path(libDPGlueHome)

    if not os.path.isfile(libDPGlueHome + "/libdpglue.a"):
        _raise_bad_env_path(libDPGlueHome + "/libdpglue.a")

glue_h = ''.join(list(filter(lambda x: len(x) > 0 and x[0] != "#", 
                             open(libDPGlueHome + '/dp_glue.h', 'r')
                             .readlines())))

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef(glue_h)

ffi_lib_name = "numba.dppy.dppy_driver._numba_dppy_bindings"

ffibuilder.set_source(
    ffi_lib_name,
    """
         #include "dp_glue.h"   // the C header of the library
    """,
    libraries=["dpglue", "OpenCL"],
    include_dirs=[libDPGlueHome],
    library_dirs=[libDPGlueHome]
)   # library name, for the linker


if __name__ == "__main__":
    # ffibuilder.emit_c_code("pybindings.c")
    ffibuilder.compile(verbose=True)
