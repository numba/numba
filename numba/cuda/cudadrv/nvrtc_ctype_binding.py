# Copyright (c) 2014-2018, NVIDIA Corporation.  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from ctypes import (
    POINTER,
    c_int,
    c_void_p,
    byref,
    c_char_p,
    c_size_t,
    cdll,
)
from ctypes.util import find_library
from enum import IntEnum


# NVRTC status codes
class nvrtcResult(IntEnum):
    NVRTC_SUCCESS = 0
    NVRTC_ERROR_OUT_OF_MEMORY = 1
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
    NVRTC_ERROR_INVALID_INPUT = 3
    NVRTC_ERROR_INVALID_PROGRAM = 4
    NVRTC_ERROR_INVALID_OPTION = 5
    NVRTC_ERROR_COMPILATION = 6
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7


def encode_str_list(str_list):
    return [s for s in str_list]


class NVRTCInterface(object):
    """
    Low-level interface to NVRTC.  This class is primarily designed for
    interfacing the high-level API with the NVRTC binary, but clients
    are free to use NVRTC directly through this class.
    """
    def __init__(self, lib_path=''):
        self._lib = None
        self._load_nvrtc_lib(lib_path)

    def _load_nvrtc_lib(self, lib_path):
        """
        Loads the NVRTC shared library, with an optional search path in
        lib_path.
        """
        def_lib_name = "nvrtc"

        if len(lib_path) == 0:
            name = def_lib_name
        else:
            name = lib_path

        found_library = find_library(name)
        self._lib = cdll.LoadLibrary(found_library)

        self._lib.nvrtcCreateProgram.argtypes = [
            POINTER(c_void_p),  # prog
            c_char_p,           # src
            c_char_p,           # name
            c_int,              # numHeaders
            POINTER(c_char_p),  # headers
            POINTER(c_char_p)   # include_names
        ]
        self._lib.nvrtcCreateProgram.restype = c_int

        self._lib.nvrtcDestroyProgram.argtypes = [
            POINTER(c_void_p)   # prog
        ]
        self._lib.nvrtcDestroyProgram.restype = c_int

        self._lib.nvrtcCompileProgram.argtypes = [
            c_void_p,           # prog
            c_int,              # numOptions
            POINTER(c_char_p)   # options
        ]
        self._lib.nvrtcCompileProgram.restype = c_int

        self._lib.nvrtcGetPTXSize.argtypes = [
            c_void_p,           # prog
            POINTER(c_size_t)   # ptxSizeRet
        ]
        self._lib.nvrtcGetPTXSize.restype = c_int

        self._lib.nvrtcGetPTX.argtypes = [
            c_void_p,           # prog
            c_char_p            # ptx
        ]
        self._lib.nvrtcGetPTX.restype = c_int

        self._lib.nvrtcGetProgramLogSize.argtypes = [
            c_void_p,           # prog
            POINTER(c_size_t)   # logSizeRet
        ]
        self._lib.nvrtcGetProgramLogSize.restype = c_int

        self._lib.nvrtcGetProgramLog.argtypes = [
            c_void_p,           # prog
            c_char_p            # log
        ]
        self._lib.nvrtcGetProgramLog.restype = c_int

        self._lib.nvrtcVersion.argtypes = [
            POINTER(c_int),     # major
            POINTER(c_int)      # minor
        ]
        self._lib.nvrtcVersion.restype = c_int

    def nvrtcCreateProgram(self, src, name, num_headers, headers,
                           include_names):
        """
        Creates and returns a new NVRTC program object.
        """
        res = c_void_p()
        headers_array = (c_char_p * num_headers)()
        headers_array[:] = encode_str_list(headers)
        include_names_array = (c_char_p * len(include_names))()
        include_names_array[:] = encode_str_list(include_names)
        code = self._lib.nvrtcCreateProgram(byref(res),
                                            c_char_p(src),
                                            c_char_p(name),
                                            len(headers),
                                            headers_array, include_names_array)

        return (nvrtcResult(code), res)

    def nvrtcDestroyProgram(self, prog):
        """
        Destroys the given NVRTC program object.
        """
        code = self._lib.nvrtcDestroyProgram(byref(prog))
        return (nvrtcResult(code),)

    def nvrtcCompileProgram(self, prog, options_len, options):
        """
        Compiles the NVRTC program object into PTX, using the provided options
        array.  See the NVRTC API documentation for accepted options.
        """

        options_array = (c_char_p * options_len)()
        options_array[:] = encode_str_list(options)
        code = self._lib.nvrtcCompileProgram(prog, options_len, options_array)
        return (nvrtcResult(code),)

    def nvrtcGetPTX(self, prog, ptx_buf):
        """
        Returns the compiled PTX for the NVRTC program object.
        """
        code = self._lib.nvrtcGetPTX(prog, ptx_buf)
        return (nvrtcResult(code),)

    def nvrtcGetPTXSize(self, prog):
        size = c_size_t()
        code = self._lib.nvrtcGetPTXSize(prog, byref(size))

        return (nvrtcResult(code), size.value)

    def nvrtcGetProgramLog(self, prog, log_buf):
        """
        Returns the log for the NVRTC program object.

        Only useful after calls to nvrtcCompileProgram or nvrtcVerifyProgram.
        """
        code = self._lib.nvrtcGetProgramLog(prog, log_buf)

        return (nvrtcResult(code),)

    def nvrtcGetProgramLogSize(self, prog):
        size = c_size_t()
        code = self._lib.nvrtcGetProgramLogSize(prog, byref(size))
        return (nvrtcResult(code), size.value)

    def nvrtcVersion(self):
        """
        Returns the loaded NVRTC library version as a (major, minor) tuple.
        """
        major = c_int()
        minor = c_int()
        code = self._lib.nvrtcVersion(byref(major), byref(minor))
        return (nvrtcResult(code), major.value, minor.value)

    def __str__(self):
        code, major, minor = self.nvrtcVersion()
        return 'NVRTC Interface (Version: %d.%d, Status: %s)' % (
            major, minor, nvrtcResult(code).name)

    def __repr__(self):
        return str(self)
