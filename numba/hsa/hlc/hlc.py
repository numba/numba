# A temporary wrapper to connect to the HLC LLVM binaries.
# Currently, connect to commandline interface.
from __future__ import print_function, absolute_import
from subprocess import check_call
import tempfile
import os

from numba import config


class CmdLine(object):
    CMD_OPT = ("$HSAILBIN/opt "
               "-O3 "
               "-gpu "
               "-whole "
               "-verify "
               "-S "
               "-o {fout} "
               "{fin}")

    CMD_VERIFY = ("$HSAILBIN/opt "
                  "-verify "
                  "-S "
                  "-o {fout} "
                  "{fin}")

    CMD_GEN_HSAIL = ("$HSAILBIN/llc -O2 "
                     "-march=hsail64 "
                     "-filetype=asm "
                     "-o {fout} "
                     "{fin}")

    CMD_LINK_BUILTINS = ("$HSAILBIN/llvm-link "
                         "-prelink-opt "
                         "-o {fout} "
                         "{fin} "
                         "-l$HSAILBIN/builtins-hsail.bc")

    def verify(self, ipath, opath):
        check_call(self.CMD_VERIFY.format(fout=opath, fin=ipath), shell=True)

    def optimize(self, ipath, opath):
        check_call(self.CMD_OPT.format(fout=opath, fin=ipath), shell=True)

    def generate_hsail(self, ipath, opath):
        check_call(self.CMD_GEN_HSAIL.format(fout=opath, fin=ipath), shell=True)

    def link_builtins(self, ipath, opath):
        check_call(self.CMD_LINK_BUILTINS.format(fout=opath, fin=ipath),
                   shell=True)


class Module(object):
    def __init__(self):
        """
        Setup
        """
        self._tmpdir = tempfile.mkdtemp()
        self._tempfiles = []
        self._linkfiles = []
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

    def load_llvm(self, llvmir):
        """
        Load LLVM with HSAIL SPIR spec
        """
        # Create temp file to store the input file
        tmp_llvm_ir, fin = self._create_temp_file("dump-llvm-ir")
        with tmp_llvm_ir:
            tmp_llvm_ir.write(llvmir.encode('ascii'))

        # Create temp file for optimization
        fout = self._track_temp_file("optimized-llvm-ir")
        self._cmd.optimize(ipath=fin, opath=fout)

        if config.DUMP_OPTIMIZED:
            with open(fout, 'rb') as fin_opt:
                print(fin_opt.read().decode('ascii'))

        self._linkfiles.append(fout)

    def finalize(self):
        """
        Finalize module and return the HSAIL code
        """
        assert not self._finalized, "Module finalized already"
        assert len(self._linkfiles) == 1, "does not support multiple modules"
        # Link library with the builtin modules
        llvmfile = self._linkfiles[0]
        linked_path = self._track_temp_file("linked-path")
        self._cmd.link_builtins(ipath=llvmfile, opath=linked_path)

        # Finalize the llvm to HSAIL
        hsail_path = self._track_temp_file("finalized-hsail")
        self._cmd.generate_hsail(ipath=linked_path, opath=hsail_path)

        self._finalized = True

        # Read HSAIL
        with open(hsail_path, 'rb') as fin:
            hsail = fin.read().decode('ascii')

        if config.DUMP_ASSEMBLY:
            print(hsail)

        return hsail
