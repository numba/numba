import sys
import os.path
import os

def get_lib_dir():
    dirname = 'DLLs' if sys.platform == 'win32' else 'lib'
    libdir = os.path.join(sys.prefix, dirname)
    return libdir


DLLNAMEMAP = {
    'linux2': ('lib', '.so'),
    'darwin': ('lib', '.dylib'),
    'win32' : ('',    '.dll')
}

def find_lib(libname, env=None):
    prefix, ext = DLLNAMEMAP[sys.platform]
    fullname = '%s%s%s' % (prefix, libname, ext)
    candidates = [
        os.path.join(get_lib_dir(), fullname),
        fullname,
        ]

    if env:
        envpath = os.environ.get(env, '')
    else:
        envpath = None

    if envpath:
        if isdir(envpath):
            envpath = os.path.join(envpath, fullname)
        return [envpath]
    else:
        return candidates
