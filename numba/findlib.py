from __future__ import print_function, absolute_import
import sys
import os
import re


def get_lib_dirs():
    """
    Anaconda specific
    """
    if sys.platform == 'win32':
        # on windows, historically `DLLs` has been used for CUDA libraries,
        # since approximately CUDA 9.2, `Library\bin` has been used.
        dirnames = ['DLLs', os.path.join('Library', 'bin')]
    else:
        dirnames = ['lib', ]
    libdirs = [os.path.join(sys.prefix, x) for x in dirnames]
    return libdirs


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
    if libdir is None:
        libdirs = get_lib_dirs()
    elif isinstance(libdir, str):
        libdirs = [libdir,]
    else:
        libdirs = list(libdir)
    files = []
    for ldir in libdirs:
        entries = os.listdir(ldir)
        candidates = [os.path.join(ldir, ent)
                      for ent in entries if pat.match(ent)]
        files.extend([c for c in candidates if os.path.isfile(c)])
    return files
