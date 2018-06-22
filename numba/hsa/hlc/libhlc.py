from __future__ import absolute_import, print_function

import os
import sys
from collections import namedtuple
from ctypes import (c_size_t, byref, c_char_p, c_void_p, Structure, CDLL,
                    POINTER, create_string_buffer, c_int, addressof,
                    c_byte)
import tempfile
import os
from numba import utils, config
from .utils import adapt_llvm_version
from .config import BUILTIN_PATH, WRAPPER_PATH


from numba.hsa.hlc.hlc import CmdLine

# the CLI tooling is needed for the linking phase at present
cli = CmdLine()


class OpaqueModuleRef(Structure):
    pass


moduleref_ptr = POINTER(OpaqueModuleRef)


def set_option(*opt):
    """
    Use this for setting debug flags to libHLC using the same options
    available to LLVM.
    E.g -debug-pass=Structure
    """
    inp = [create_string_buffer(x.encode('ascii')) for x in (('libhlc',) + opt)]
    argc = len(inp)
    argv = (c_char_p * argc)()
    for i in range(argc):
        argv[i] = addressof(inp[i])
    hlc.HLC_SetCommandLineOption(argc, byref(argv))


class Error(Exception):
    pass


class HLC(object):
    """
    LibHLC wrapper interface
    """
    hlc = None

    def __init__(self):
        # Lazily load the libHLC library
        if self.hlc is None:
            try:
                hlc = CDLL(os.path.join(sys.prefix, 'lib', 'libHLC.so'))
            except OSError:
                raise ImportError("libHLC.so cannot be found.  Please install the libhlc "
                                  "package by: conda install -c numba libhlc")

            else:
                hlc.HLC_ParseModule.restype = moduleref_ptr
                hlc.HLC_ParseBitcode.restype = moduleref_ptr
                hlc.HLC_ModuleEmitBRIG.restype = c_size_t
                hlc.HLC_Initialize()
                utils.finalize(hlc, hlc.HLC_Finalize)

                hlc.HLC_SetCommandLineOption.argtypes = [
                    c_int,
                    c_void_p,
                ]

                type(self).hlc = hlc

    def parse_assembly(self, ir):
        if isinstance(ir, str):
            ir = ir.encode("latin1")
        buf = create_string_buffer(ir)
        mod = self.hlc.HLC_ParseModule(buf)
        if not mod:
            raise Error("Failed to parse assembly")
        return mod

    def parse_bitcode(self, bitcode):
        buf = create_string_buffer(bitcode, len(bitcode))
        mod = self.hlc.HLC_ParseBitcode(buf, c_size_t(len(bitcode)))
        if not mod:
            raise Error("Failed to parse bitcode")
        return mod

    def optimize(self, mod, opt=3, size=0, verify=1):
        if not self.hlc.HLC_ModuleOptimize(mod, int(opt), int(size), int(verify)):
            raise Error("Failed to optimize module")

    def link(self, dst, src):
        if not self.hlc.HLC_ModuleLinkIn(dst, src):
            raise Error("Failed to link modules")

    def to_hsail(self, mod, opt=2):
        buf = c_char_p(0)
        if not self.hlc.HLC_ModuleEmitHSAIL(mod, int(opt), byref(buf)):
            raise Error("Failed to emit HSAIL")
        ret = buf.value.decode("latin1")
        self.hlc.HLC_DisposeString(buf)
        return ret

    def _link_brig(self, upbrig_loc, patchedbrig_loc):
        cli.link_brig(upbrig_loc, patchedbrig_loc)

    def to_brig(self, mod, opt=2):
        bufptr = c_void_p(0)
        size = self.hlc.HLC_ModuleEmitBRIG(mod, int(opt), byref(bufptr))
        if not size:
            raise Error("Failed to emit BRIG")
        buf = (c_byte * size).from_address(bufptr.value)
        try:
            buffer
        except NameError:
            ret = bytes(buf)
        else:
            ret = bytes(buffer(buf))
        self.hlc.HLC_DisposeString(buf)
        # Now we have an ELF, this needs patching with ld.lld which doesn't
        # have an API. So we write out `ret` to a temporary file, then call
        # the ld.lld ELF linker main() on it to generate a patched ELF
        # temporary file output, which we read back in.

        # tmpdir
        tmpdir = tempfile.mkdtemp()
        tmp_files = []

        # write out unpatched BRIG
        upbrig_file = "unpatched.brig"
        upbrig_loc = os.path.join(tmpdir, upbrig_file)
        with open(upbrig_loc, "wb") as up_brig_fobj:
            up_brig_fobj.write(ret)
            up_brig_fobj.close()
            tmp_files.append(upbrig_loc)

        # record the location of the patched ELF
        patchedbrig_file = "patched.brig"
        patchedbrig_loc = os.path.join(tmpdir, patchedbrig_file)

        # call out to ld.lld to patch
        self._link_brig(upbrig_loc, patchedbrig_loc)

        # read back in brig temporary.
        with open(patchedbrig_loc, "rb") as p_brig_fobj:
            patchedBrig = p_brig_fobj.read()
            p_brig_fobj.close()
            tmp_files.append(patchedbrig_loc)

#        # Remove all temporary files
#        for afile in tmp_files:
#            os.unlink(afile)
#        # Remove directory
#        os.rmdir(tmpdir)

        return patchedBrig

    def to_string(self, mod):
        buf = c_char_p(0)
        self.hlc.HLC_ModulePrint(mod, byref(buf))
        ret = buf.value.decode("latin1")
        self.hlc.HLC_DisposeString(buf)
        return ret

    def destroy_module(self, mod):
        self.hlc.HLC_ModuleDestroy(mod)


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

    def generateGCN(self):
        """
        Finalize module and return the HSAIL code
        """
        assert not self._finalized, "Module finalized already"

        # Link dependencies
        main = self._llvm_modules[0]
        for dep in self._llvm_modules[1:]:
            self._hlc.link(main, dep)

        # Link library with the builtin modules
        with open(BUILTIN_PATH, 'rb') as builtin_fin:
            builtin_buf1 = builtin_fin.read()
        builtin_mod1 = self._hlc.parse_bitcode(builtin_buf1)
        self._hlc.link(main, builtin_mod1)

        # link library with the builtin hsail-amdgpu-wrapper.ll
        with open(WRAPPER_PATH, 'rb') as builtin_fin:
            builtin_buf2 = builtin_fin.read()
        builtin_mod2 = self._hlc.parse_assembly(builtin_buf2)
        self._hlc.link(main, builtin_mod2)

        # Optimize
        self._hlc.optimize(main)

        if config.DUMP_OPTIMIZED:
            print(self._hlc.to_string(main))

        # create HSAIL
        hsail = self._hlc.to_hsail(main)

        # Finalize the llvm to BRIG
        brig = self._hlc.to_brig(main)

        self._finalized = True

        # Clean up main; other modules are destroyed at linking
        self._hlc.destroy_module(main)

        if config.DUMP_ASSEMBLY:
            print(hsail)

        return namedtuple('FinalizerResult', ['hsail', 'brig'])(hsail, brig)
