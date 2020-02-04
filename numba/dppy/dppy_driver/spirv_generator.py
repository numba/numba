# A wrapper to connect to the SPIR-V binaries (Tools, Translator).
# Currently, connect to commandline interface.
from __future__ import print_function, absolute_import
import sys
import os
from subprocess import check_call, CalledProcessError, call
import tempfile

from numba import config

BAD_spirv_tools_home_PATH_ERRMSG = """
SPIRV_TOOLS_HOME is set to '{0}' which is not a valid path. Set this to the
location where you installed SPIRV-Tools.
"""

BAD_LLVM_HOME_PATH_ERRMSG = """
LLVM_HOME is set to '{0}' which is not a valid path. Set this to the
location where the LLVM toolchain is installed.
"""

spirv_tools_home = os.environ.get('SPIRV_TOOLS_HOME', None)
llvm_home = os.environ.get('LLVM_HOME', None)


def _raise_bad_env_path(msg, path, extra=None):
    error_message = msg.format(path)
    if extra is not None:
        error_message += extra
    raise ValueError(error_message)


if spirv_tools_home is None:
    raise ValueError("FATAL: Correctly set the SPIRV_TOOLS_HOME environment "
                     "variable.")

if spirv_tools_home is not None:
    try:
        os.path.abspath(spirv_tools_home)
    except ValueError:
        _raise_bad_env_path(BAD_spirv_tools_home_PATH_ERRMSG, spirv_tools_home)

    if not os.path.isfile(spirv_tools_home + "/bin/spirv-val"):
        _raise_bad_env_path(BAD_spirv_tools_home_PATH_ERRMSG,
                            spirv_tools_home + "/bin/spirv-val")

if llvm_home is None:
    raise ValueError("FATAL: Correctly set the LLVM_HOME environment variable.")

if llvm_home is not None:
    try:
        os.path.abspath(llvm_home)
    except ValueError:
        _raise_bad_env_path(BAD_LLVM_HOME_PATH_ERRMSG, llvm_home)

    if not os.path.isfile(llvm_home + "/bin/opt"):
        _raise_bad_env_path(BAD_LLVM_HOME_PATH_ERRMSG,
                            llvm_home + "/bin/opt")

_real_check_call = check_call


def check_call(*args, **kwargs):
    return _real_check_call(*args, **kwargs)

class CmdLine(object):

    def disassemble(self, ipath, opath):
        check_call([
            spirv_tools_home + "/bin/spirv-dis",
            # "--no-indent",
            # "--no-header",
            # "--raw-id",
            # "--offsets",
            "-o",
            opath,
            ipath])

    def validate(self, ipath):
        check_call([spirv_tools_home + "/bin/spirv-val",ipath])

    def optimize(self, ipath, opath):
        check_call([
            spirv_tools_home + "/bin/spirv-opt",
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
        check_call([llvm_home+"/bin/opt","-O3","-o",ipath+'.bc',ipath])
        check_call([llvm_home + "/bin/llvm-spirv","-o",opath,ipath+'.bc'])
        os.unlink(ipath + '.bc')


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

    # def _preprocess(self, llvmir):
    #    return adapt_llvm_version(llvmir)

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

        # Validate the SPIR-V code
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

def llvm_to_spirv(bitcode):
    mod = Module()
    mod.load_llvm(bitcode)
    return mod.finalize()
