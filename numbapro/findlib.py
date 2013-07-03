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
        if os.path.isdir(envpath):
            envpath = os.path.join(envpath, fullname)
        return [envpath]
    else:
        return candidates

def test():
    failed = False
    libs = 'cublas cusparse cufft curand nvvm'.split()
    for lib in libs:
        cand = find_lib(lib)
        print 'Finding', lib
        for path in cand:
            if os.path.isfile(path):
                print '\tlocated at', path
                break
        else:
            print 'can\'t open lib'
            failed = True
    bcs = ['libdevice.compute_20.bc', 'libdevice.compute_30.bc',
           'libdevice.compute_35.bc']
    bcdir = get_lib_dir()
    listing = os.listdir(bcdir)
    print 'In', bcdir
    for bc in bcs:
        print '\tfinding', bc,

        if bc not in listing:
            print '\tcan\'t open %s' % bc
            failed = True
        else:
            print '\tok'
    return not failed
