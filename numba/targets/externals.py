"""
Register external C functions necessary for Numba code generation.
"""

import sys

import llvmlite.binding as ll

from numba import utils
from numba import _helperlib, _npymath_exports
from . import intrinsics


def _add_missing_symbol(symbol, addr):
    """Add missing symbol into LLVM internal symtab
    """
    if not ll.address_of_symbol(symbol):
        ll.add_symbol(symbol, addr)


def _get_msvcrt_symbol(symbol):
    """
    Under Windows, look up a symbol inside the C runtime
    and return the raw pointer value as an integer.
    """
    from ctypes import cdll, cast, c_void_p
    f = getattr(cdll.msvcrt, symbol)
    return cast(f, c_void_p).value


class _Installer(object):

    _installed = False

    def install(self):
        """
        Install the functions into LLVM.  This only needs to be done once,
        as the mappings are persistent during the process lifetime.
        """
        if not self._installed:
            self._do_install()
            self._installed = True


class _ExternalMathFunctions(_Installer):
    """
    Map the math functions from the C runtime library into the LLVM
    execution environment.
    """

    def _do_install(self):
        is32bit = utils.MACHINE_BITS == 32
        c_helpers = _helperlib.c_helpers
        for name in ['cpow', 'sdiv', 'srem', 'udiv', 'urem']:
            ll.add_symbol("numba.math.%s" % name, c_helpers[name])

        if sys.platform.startswith('win32') and is32bit:
            # For Windows XP _ftol2 is not defined, we will just use
            # _ftol as a replacement.
            # On Windows 7, this is not necessary but will work anyway.
            ftol = _get_msvcrt_symbol("_ftol")
            _add_missing_symbol("_ftol2", ftol)

        elif sys.platform.startswith('linux') and is32bit:
            _add_missing_symbol("__fixunsdfdi", c_helpers["fptoui"])
            _add_missing_symbol("__fixunssfdi", c_helpers["fptouif"])

        if is32bit:
            _add_missing_symbol("__multi3", c_helpers["multi3"])

        # List available C-math
        for fname in intrinsics.INTR_MATH:
            # Force binding from CPython's C runtime library.
            # (under Windows, different versions of the C runtime can
            #  be loaded at the same time, for example msvcrt100 by
            #  CPython and msvcrt120 by LLVM)
            ll.add_symbol(fname, c_helpers[fname])


class _ExternalNumpyFunctions(_Installer):
    """
    Map Numpy's C math functions into the LLVM execution environment.
    """

    def _do_install(self):
        # add the symbols for numpy math to the execution environment.
        for sym in _npymath_exports.symbols:
            ll.add_symbol(*sym)


c_math_functions = _ExternalMathFunctions()
c_numpy_functions = _ExternalNumpyFunctions()
