# A wrapper to connect to the SPIR 2.0 binaries (Tools, Translator).
# Currently, connect to commandline interface.
import sys, os
from subprocess import check_call, CalledProcessError
import tempfile
import re

from numba import config

os.environ['SPIRVDIR'] = os.environ.get('SPIRVDIR', '/opt/spirv')

_real_check_call = check_call

def check_call(*args, **kwargs):
    return _real_check_call(*args, **kwargs)


class CmdLine(object):
    CMD_AS = ("$SPIRVDIR/llvm-as "
              "-o {fout} "
              "{fin}")

    def assemble(self, ipath, opath):
        check_call(self.CMD_AS.format(fout=opath, fin=ipath), shell=True)


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

    def _preprocess(self, llvmir):
        return adapt_llvm_version(llvmir)

    def load_llvm(self, llvmir):
        """
        Load SPIR 2.0 IR based on LLVM 3.4
        """
        # Preprocess LLVM IR, from 4.0 to 3.4, the
        llvmir = self._preprocess(llvmir)

        # Create temp file to store the input file
        tmp_llvm_ir, llvm_path = self._create_temp_file("llvm-4.0-spir")
        with tmp_llvm_ir:
            tmp_llvm_ir.write(llvmir)

        self._llvmfile = llvm_path

    def finalize(self):
        """
        Finalize module and return the SPIR-V code
        """
        assert not self._finalized, "Module finalized already"

        # Obtain binary SPIR 2.0 with the LLVM assembler 3.4
        spir2_path = self._track_temp_file("binary-spir2")
        self._cmd.assemble(ipath=self._llvmfile, opath=spir2_path)

        # Read and return binary SPIR 2.0
        with open(spir2_path, 'rb') as fin:
            spir2 = fin.read()

        self._finalized = True

        return spir2


# Public llvm_to_spir2 function ###############################################

def llvm_to_spir2(bitcode):
    mod = Module()
    mod.load_llvm(bitcode)
    return mod.finalize()


# ========= adapt llvm 4.0 IR to llvm 3.4

re_load_instr = re.compile(r"load\s+[a-zA-Z0-9 _\[\]*x]+\s*,")
re_gep_instr = re.compile(r"getelementptr\s+(?:inbounds)?[a-zA-Z0-9 _\[\]*x]+\s*,")

re_metadata_def = re.compile(r"\!\d+\s*=")
re_metadata_correct_usage = re.compile(r"metadata\s*\![{'\"]")
re_metadata_ref = re.compile(r"\!\d+")


def add_metadata_type(ir):
    """
    Rewrite metadata since llvm3.6 dropped the "metadata" type prefix.
    """
    buf = []
    for line in ir.splitlines():
        # If the line is a metadata
        if re_metadata_def.match(line):
            # Does not contain any correct usage (Maybe already fixed)
            if None is re_metadata_correct_usage.search(line):
                line = line.replace('!{', 'metadata !{')
                line = line.replace('!"', 'metadata !"')

                def sub_metadata(m):
                    return "metadata {0}".format(m.group(0))

                line = re_metadata_ref.sub(sub_metadata, line)
                line = line.lstrip('metadata ')
        buf.append(line)

    return '\n'.join(buf)


def correct_load_gep(ir):
    """
    LLVM < 3.7 has a different syntax for load and gep instr.
    """
    ir = re_load_instr.sub(r"load", ir)
    return re_gep_instr.sub(r"getelementptr", ir)


def adapt_llvm_version(ir):
    """
    Adapt the LLVM IR to match the syntax required by SPIR 2.0
    """
    ir = ir.replace('source_filename', '; source_filename')
    ir = ir.replace('local_unnamed_addr','')
    ir = ir.replace('nonnull', '')
    ir = ir.replace('norecurse', '')
    ir = correct_load_gep(ir)
    return add_metadata_type(ir)
