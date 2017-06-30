# A wrapper to connect to the SPIR-V binaries (Tools, Translator).
# Currently, connect to commandline interface.
from __future__ import print_function, absolute_import
import sys, os
from subprocess import check_call, CalledProcessError
import tempfile

from numba import config

os.environ['SPIRVDIR'] = os.environ.get('SPIRVDIR', '/opt/spirv')

_real_check_call = check_call

def check_call(*args, **kwargs):
    return _real_check_call(*args, **kwargs)


class CmdLine(object):
    CMD_AS = ("$SPIRVDIR/spirv-as "
              # "--preserve-numeric-ids"
              # "--target_env {vulkan1.0|spv1.0|spv1.1|spv1.2}"
              "-o {fout} "
              "{fin}")

    CMD_DIS = ("$SPIRVDIR/spirv-dis "
               # "--no-indent"
               # "--no-header"
               # "--raw-id"
               # "--offsets"
               "-o {fout} "
               "{fin}")

    CMD_VAL = ("$SPIRVDIR/spirv-val "
               # "--target_env {vulkan1.0|spv1.0|spv1.1|spv1.2}"
               "{fin}")

    CMD_OPT = ("$SPIRVDIR/spirv-opt "
               # "--strip-debug"
               # "--freeze-spec-const"
               # "--eliminate-dead-const"
               # "--fold-spec-const-op-composite"
               # "--set-spec-const-default-value '<spec id>:<default value> ...'"
               # "--unify-const"
               # "--inline-entry-points-exhaustive"
               # "--flatten-decorations"
               # "--compact-ids"
               "-o {fout} "
               "{fin}")

    CMD_GEN = ("$SPIRVDIR/llc "
               "-march=spirv64 "
               "-o {fout} "
               "{fin}")

    def assemble(self, ipath, opath):
        check_call(self.CMD_AS.format(fout=opath, fin=ipath), shell=True)

    def disassemble(self, ipath, opath):
        check_call(self.CMD_DIS.format(fout=opath, fin=ipath), shell=True)

    def validate(self, ipath):
        check_call(self.CMD_VAL.format(fin=ipath), shell=True)

    def optimize(self, ipath, opath):
        check_call(self.CMD_OPT.format(fout=opath, fin=ipath), shell=True)

    def generate(self, ipath, opath):
        check_call(self.CMD_GEN.format(fout=opath, fin=ipath), shell=True)


class Module(object):
    def __init__(self):
        """
        Setup
        """
        self._tmpdir = tempfile.mkdtemp()
        self._tempfiles = []
        self._cmd = CmdLine()
        self._finalized = False

    def __del__(self):
        self.close()

    def close(self):
        # Remove all temporary files
        for afile in self._tempfiles:
            os.unlink(afile)
        # Remove directory
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

    #def _preprocess(self, llvmir):
    #    return adapt_llvm_version(llvmir)

    def load_llvm(self, llvmir):
        """
        Load LLVM with "SPIR-V friendly" SPIR 2.0 spec
        """
        # Create temp file to store the input file
        tmp_llvm_ir, llvm_path = self._create_temp_file("llvm-friendly-spir")
        with tmp_llvm_ir:
            tmp_llvm_ir.write(llvmir)

        self._llvmfile = llvm_path

    def finalize(self):
        """
        Finalize module and return the SPIR-V code
        """
        assert not self._finalized, "Module finalized already"

        # Generate SPIR-V from "friendly" LLVM-based SPIR 2.0
        spirv_path = self._track_temp_file("generated-spirv")
        self._cmd.generate(ipath=self._llvmfile, opath=spirv_path)

        # Validate the SPIR-V code
        try:
            self._cmd.validate(ipath=spirv_path)
        except CalledProcessError:
            pass
        else:
            # Optimize SPIR-V code
            opt_path = self._track_temp_file("optimized-spirv")
            self._cmd.optimize(ipath=spirv_path, opath=opt_path)

            # Disassemble optimized SPIR-V code
            dis_path = self._track_temp_file("disassembled-spirv")
            self._cmd.disassemble(ipath=opt_path, opath=dis_path)

            if config.DUMP_ASSEMBLY:
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

def llvm_to_spirv(bitcode):
    mod = Module()
    mod.load_llvm(bitcode)
    return mod.finalize()
