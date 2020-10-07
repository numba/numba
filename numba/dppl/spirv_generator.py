# A wrapper to connect to the SPIR-V binaries (Tools, Translator).
# Currently, connect to commandline interface.
from __future__ import print_function, absolute_import
import sys
import os
from subprocess import check_call, CalledProcessError, call
import tempfile

from numba import config
from numba.dppl.target import LINK_ATOMIC


def _raise_bad_env_path(msg, path, extra=None):
    error_message = msg.format(path)
    if extra is not None:
        error_message += extra
    raise ValueError(error_message)

_real_check_call = check_call

def check_call(*args, **kwargs):
    #print("check_call:", *args, **kwargs)
    return _real_check_call(*args, **kwargs)

class CmdLine(object):

    def disassemble(self, ipath, opath):
        check_call([
            "spirv-dis",
            # "--no-indent",
            # "--no-header",
            # "--raw-id",
            # "--offsets",
            "-o",
            opath,
            ipath])

    def validate(self, ipath):
        check_call(["spirv-val",ipath])

    def optimize(self, ipath, opath):
        check_call([
            "spirv-opt",
            # "--strip-debug",
            # "--freeze-spec-const",
            # "--eliminate-dead-const",
            # "--fold-spec-const-op-composite",
            # "--set-spec-const-default-value '<spec id>:<default value> ...'",
            # "--unify-const",
            # "--inline-entry-points-exhaustive",
            # "--flatten-decorations",
            # "--compact-ids",
            "-o",
            opath,
            ipath])

    def generate(self, ipath, opath):
        # DRD : Temporary hack to get SPIR-V code generation to work.
        # The opt step is needed for:
        #     a) generate a bitcode file from the text IR file
        #     b) hoist all allocas to the enty block of the module
        check_call(["opt","-O1","-o",ipath+'.bc',ipath])
        check_call(["llvm-spirv","-o",opath,ipath+'.bc'])
        if config.SAVE_DPPL_IR_FILES == 0:
            os.unlink(ipath + '.bc')

    def link(self, opath, binaries):
        params = ["spirv-link","--allow-partial-linkage","-o", opath]
        params.extend(binaries)

        check_call(params)

class Module(object):
    def __init__(self, context):
        """
        Setup
        """
        self._tmpdir = tempfile.mkdtemp()
        self._tempfiles = []
        self._cmd = CmdLine()
        self._finalized = False
        self.context = context

    def __del__(self):
        # Remove all temporary files
        for afile in self._tempfiles:
            if config.SAVE_DPPL_IR_FILES != 0:
                print(afile)
            else:
                os.unlink(afile)
        # Remove directory
        if config.SAVE_DPPL_IR_FILES == 0:
            os.rmdir(self._tmpdir)

    def _create_temp_file(self, name, mode='wb'):
        path = self._track_temp_file(name)
        fobj = open(path, mode=mode)
        return fobj, path

    def _track_temp_file(self, name):
        path = os.path.join(self._tmpdir,
                            "{0}-{1}".format(len(self._tempfiles), name))
        self._tempfiles.append(path)
        return path

    def load_llvm(self, llvmir):
        """
        Load LLVM with "SPIR-V friendly" SPIR 2.0 spec
        """
        # Create temp file to store the input file
        tmp_llvm_ir, llvm_path = self._create_temp_file("llvm-friendly-spir")
        with tmp_llvm_ir:
            tmp_llvm_ir.write(llvmir.encode())

        self._llvmfile = llvm_path

    def finalize(self):
        """
        Finalize module and return the SPIR-V code
        """
        assert not self._finalized, "Module finalized already"

        # Generate SPIR-V from "friendly" LLVM-based SPIR 2.0
        spirv_path = self._track_temp_file("generated-spirv")
        self._cmd.generate(ipath=self._llvmfile, opath=spirv_path)

        binary_paths = [spirv_path]
        for key in list(self.context.link_binaries.keys()):
            del self.context.link_binaries[key]
            if key == LINK_ATOMIC:
                from .ocl.atomics import get_atomic_spirv_path
                binary_paths.append(get_atomic_spirv_path())

        if len(binary_paths) > 1:
            spirv_path = self._track_temp_file("linked-spirv")
            self._cmd.link(spirv_path, binary_paths)

        # Validate the SPIR-V code
        if config.SPIRV_VAL == 1:
            try:
                self._cmd.validate(ipath=spirv_path)
            except CalledProcessError:
                print("SPIR-V Validation failed...")
                pass
            else:
                # Optimize SPIR-V code
                opt_path = self._track_temp_file("optimized-spirv")
                self._cmd.optimize(ipath=spirv_path, opath=opt_path)

                if config.DUMP_ASSEMBLY:
                    # Disassemble optimized SPIR-V code
                    dis_path = self._track_temp_file("disassembled-spirv")
                    self._cmd.disassemble(ipath=opt_path, opath=dis_path)
                    with open(dis_path, 'rb') as fin_opt:
                        print("ASSEMBLY".center(80, "-"))
                        print(fin_opt.read())
                        print("".center(80, "="))

        # Read and return final SPIR-V (not optimized!)
        with open(spirv_path, 'rb') as fin:
            spirv = fin.read()

        self._finalized = True

        return spirv


# Public llvm_to_spirv function ###############################################

def llvm_to_spirv(context, bitcode):
    mod = Module(context)
    mod.load_llvm(bitcode)
    return mod.finalize()
