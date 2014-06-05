from __future__ import print_function, absolute_import
import sys
import os
import re


def get_lib_dir():
    """
    Anaconda specific
    """
    dirname = 'DLLs' if sys.platform == 'win32' else 'lib'
    libdir = os.path.join(sys.prefix, dirname)
    return libdir


DLLNAMEMAP = {
    'linux': r'lib%(name)s\.so\.%(ver)s$',
    'linux2': r'lib%(name)s\.so\.%(ver)s$',
    'darwin': r'lib%(name)s\.%(ver)s\.dylib$',
    'win32': r'%(name)s%(ver)s\.dll$',
}

RE_VER = r'[0-9]*([_\.][0-9]+)*'


def find_lib(libname, libdir=None, platform=None):
    platform = platform or sys.platform
    pat = DLLNAMEMAP[platform] % {"name": libname, "ver": RE_VER}
    regex = re.compile(pat)
    return find_file(regex, libdir)


def find_file(pat, libdir=None):
    libdir = libdir or get_lib_dir()
    entries = os.listdir(libdir)
    candidates = [os.path.join(libdir, ent)
                  for ent in entries if pat.match(ent)]
    return [c for c in candidates if os.path.isfile(c)]
