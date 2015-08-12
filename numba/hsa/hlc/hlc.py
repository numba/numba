# A temporary wrapper to connect to the HLC LLVM binaries.
# Currently, connect to commandline interface.
from __future__ import print_function, absolute_import
import sys
from subprocess import check_call
import tempfile
import os
from collections import namedtuple

from numba import config
from .utils import adapt_llvm_version
from .config import BUILTIN_PATH

_real_check_call = check_call


def check_call(*args, **kwargs):
    print('CMD: ' + ';'.join(args), file=sys.stdout)
    return _real_check_call(*args, **kwargs)


class CmdLine(object):
    CMD_OPT = ("$HSAILBIN/opt "
               "-O3 "
               # "-gpu "
               # "-whole "
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

    CMD_GEN_BRIG = ("$HSAILBIN/llc -O2 "
                    "-march=hsail64 "
                    "-filetype=obj "
                    "-o {fout} "
                    "{fin}")

    CMD_LINK_BUILTINS = ("$HSAILBIN/llvm-link "
                         # "-prelink-opt "
                         "-S "
                         "-o {fout} "
                         "{fin} "
                         "{lib}")

    CMD_LINK_LIBS = ("$HSAILBIN/llvm-link "
                     # "-prelink-opt "
                     "-S "
                     "-o {fout} "
                     "{fin} ")

    def verify(self, ipath, opath):
        check_call(self.CMD_VERIFY.format(fout=opath, fin=ipath), shell=True)

    def optimize(self, ipath, opath):
        check_call(self.CMD_OPT.format(fout=opath, fin=ipath), shell=True)

    def generate_hsail(self, ipath, opath):
        check_call(self.CMD_GEN_HSAIL.format(fout=opath, fin=ipath), shell=True)

    def generate_brig(self, ipath, opath):
        check_call(self.CMD_GEN_BRIG.format(fout=opath, fin=ipath), shell=True)

    def link_builtins(self, ipath, opath):
        cmd = self.CMD_LINK_BUILTINS.format(fout=opath, fin=ipath,
                                            lib=BUILTIN_PATH)
        check_call(cmd, shell=True)

    def link_libs(self, ipath, libpaths, opath):
        cmdline = self.CMD_LINK_LIBS.format(fout=opath, fin=ipath)
        cmdline += ' '.join(["{0}".format(lib) for lib in libpaths])
        check_call(cmdline, shell=True)


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
        return
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
        Load LLVM with HSAIL SPIR spec
        """
        # Preprocess LLVM IR
        # Because HLC does not handle dot in LLVM variable names
        llvmir = self._preprocess(llvmir)

        # Create temp file to store the input file
        tmp_llvm_ir, fin = self._create_temp_file("dump-llvm-ir")
        with tmp_llvm_ir:
            tmp_llvm_ir.write(llvmir.encode('ascii'))

        # Create temp file for optimization
        fout = self._track_temp_file("verified-llvm-ir")
        self._cmd.verify(ipath=fin, opath=fout)

        if config.DUMP_OPTIMIZED:
            with open(fout, 'rb') as fin_opt:
                print(fin_opt.read().decode('ascii'))

        self._linkfiles.append(fout)

    def finalize(self):
        """
        Finalize module and return the HSAIL code
        """
        assert not self._finalized, "Module finalized already"

        # Link dependencies libraries
        llvmfile = self._linkfiles[0]
        pre_builtin_path = self._track_temp_file("link-dep")
        libpaths = self._linkfiles[1:]
        self._cmd.link_libs(ipath=llvmfile, libpaths=libpaths,
                            opath=pre_builtin_path)

        # Link library with the builtin modules
        linked_path = self._track_temp_file("linked-path")
        self._cmd.link_builtins(ipath=pre_builtin_path, opath=linked_path)

        # Optimize
        opt_path = self._track_temp_file("optimized-llvm-ir")
        self._cmd.optimize(ipath=linked_path, opath=opt_path)

        if config.DUMP_OPTIMIZED:
            with open(opt_path, 'rb') as fin:
                print(fin.read().decode('ascii'))

        # Finalize the llvm to HSAIL
        hsail_path = self._track_temp_file("finalized-hsail")
        self._cmd.generate_hsail(ipath=opt_path, opath=hsail_path)

        # Finalize the llvm to BRIG
        brig_path = self._track_temp_file("finalized-brig")
        self._cmd.generate_brig(ipath=opt_path, opath=brig_path)

        self._finalized = True

        # Read HSAIL
        with open(hsail_path, 'rb') as fin:
            hsail = fin.read().decode('ascii')

        # Read BRIG
        with open(brig_path, 'rb') as fin:
            brig = fin.read()

        if config.DUMP_ASSEMBLY:
            print(hsail)

        return namedtuple('FinalizerResult', ['hsail', 'brig'])(hsail, brig)
