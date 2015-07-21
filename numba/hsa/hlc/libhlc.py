from __future__ import absolute_import, print_function

from ctypes import (c_size_t, byref, c_char_p, c_void_p, Structure, CDLL,
                    POINTER, create_string_buffer)
import os
from collections import namedtuple

from numba import utils, config
from .utils import adapt_llvm_version


class OpaqueModuleRef(Structure):
    pass


moduleref_ptr = POINTER(OpaqueModuleRef)

hlc = CDLL('/home/sklam/dev/libHLC/libHLC.so')
hlc.HLC_ParseModule.restype = moduleref_ptr
hlc.HLC_ModuleEmitBRIG.restype = c_size_t
hlc.HLC_Initialize()
utils.finalize(hlc, lambda: hlc.HLC_Finalize())


class Error(Exception):
    pass


class HLC(object):
    """
    LibHLC wrapper interface
    """

    def parse_assembly(self, ir):
        if isinstance(ir, str):
            ir = ir.encode("latin1")
        buf = create_string_buffer(ir)
        mod = hlc.HLC_ParseModule(buf)
        if not mod:
            raise Error("Failed to parse assembly")
        return mod

    def parse_bitcode(self, bitcode):
        buf = create_string_buffer(bitcode, len(bitcode))
        mod = hlc.HLC_ParseBitcode(buf, c_size_t(len(bitcode)))
        if not mod:
            raise Error("Failed to parse bitcode")
        return mod

    def optimize(self, mod, opt=3, size=0, verify=1):
        if not hlc.HLC_ModuleOptimize(mod, int(opt), int(size), int(verify)):
            raise Error("Failed to optimize module")

    def link(self, dst, src):
        if not hlc.HLC_ModuleLinkIn(dst, src):
            raise Error("Failed to link modules")

    def to_hsail(self, mod, opt=3):
        buf = c_char_p(0)
        if not hlc.HLC_ModuleEmitHSAIL(mod, int(opt), byref(buf)):
            raise Error("Failed to emit HSAIL")
        ret = buf.value.decode("latin1")
        hlc.HLC_DisposeString(buf)
        return ret

    def to_brig(self, mod, opt=3):
        bufptr = c_void_p(0)
        size = hlc.HLC_ModuleEmitBRIG(mod, int(opt), byref(bufptr))
        if not size:
            raise Error("Failed to emit BRIG")
        buf = (c_char_p * size).from_address(bufptr.value)
        ret = bytes(buf)
        hlc.HLC_DisposeString(buf)
        return ret

    def to_string(self, mod):
        buf = c_char_p(0)
        if not hlc.HLC_ModulePrint(mod, byref(buf)):
            raise Error("Failed to print module")
        ret = buf.value.decode("latin1")
        hlc.HLC_DisposeString(buf)
        return ret

    def destroy_module(self, mod):
        hlc.HLC_ModuleDestroy(mod)


os.environ['HSAILBIN'] = os.environ.get('HSAILBIN', '/opt/amd/bin')

BUILTIN_PATH = "{0}/builtins-hsail.opt.bc".format(os.environ['HSAILBIN'])


class Module(object):
    def __init__(self):
        self._llvm_modules = []
        self._hlc = HLC()
        self._finalized = False

    def _preprocess(self, llvmir):
        return adapt_llvm_version(llvmir)

    def load_llvm(self, llvmir):
        """
        Load LLVM with HSAIL SPIR spec
        """
        # Preprocess LLVM IR
        # Because HLC does not handle dot in LLVM variable names
        llvmir = self._preprocess(llvmir)

        mod = self._hlc.parse_assembly(llvmir)

        if config.DUMP_OPTIMIZED:
            print(self._hlc.to_string(mod))

        self._llvm_modules.append(mod)

    def finalize(self):
        """
        Finalize module and return the HSAIL code
        """
        assert not self._finalized, "Module finalized already"

        # Link dependencies libraries
        main = self._llvm_modules[0]
        for dep in self._llvm_modules[1:]:
            self._hlc.link(main, dep)

        # Link library with the builtin modules
        with open(BUILTIN_PATH, 'rb') as builtin_fin:
            builtin_buf = builtin_fin.read()
        builtin_mod = self._hlc.parse_bitcode(builtin_buf)
        self._hlc.link(main, builtin_mod)

        # Optimize
        self._hlc.optimize(main)

        if config.DUMP_OPTIMIZED:
            print(self._hlc.to_string(main))

        # Finalize the llvm to HSAIL
        hsail = self._hlc.to_hsail(main)

        # Finalize the llvm to BRIG
        brig = self._hlc.to_brig(main)

        self._finalized = True

        # Clean up main; other modules are destroyed at linking
        self._hlc.destroy_module(main)

        if config.DUMP_ASSEMBLY:
            print(hsail)

        return namedtuple('FinalizerResult', ['hsail', 'brig'])(hsail, brig)
