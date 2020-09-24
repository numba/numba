"""Cudatoolkit libraries lookup utilities.

Cudatoolkit libraries can be available via either:

- the `cudatoolkit` conda package,
- a user supplied location from CUDA_HOME,
- a system wide location,
- package-specific locations (e.g. the Debian NVIDIA packages),
- or can be discovered by the system loader.
"""

import os
import sys
import ctypes

from numba.misc.findlib import find_lib
from numba.cuda.cuda_paths import get_cuda_paths

if sys.platform == 'win32':
    _dllnamepattern = '%s.dll'
elif sys.platform == 'darwin':
    _dllnamepattern = 'lib%s.dylib'
else:
    _dllnamepattern = 'lib%s.so'


def get_libdevice(arch):
    d = get_cuda_paths()
    paths = d['libdevice'].info
    return paths.get(arch, paths.get(None))


def open_libdevice(arch):
    with open(get_libdevice(arch), 'rb') as bcfile:
        return bcfile.read()


def get_cudalib(lib, platform=None):
    """
    Find the path of a CUDA library based on a search of known locations. If
    the search fails, return a generic filename for the library (e.g.
    'libnvvm.so' for 'nvvm') so that we may attempt to load it using the system
    loader's search mechanism.
    """
    if lib == 'nvvm':
        return get_cuda_paths()['nvvm'].info or _dllnamepattern % 'nvvm'
    else:
        libdir = get_cuda_paths()['cudalib_dir'].info

    candidates = find_lib(lib, libdir, platform)
    return max(candidates) if candidates else _dllnamepattern % lib


def open_cudalib(lib):
    path = get_cudalib(lib)
    return ctypes.CDLL(path)


def _get_source_variable(lib):
    if lib == 'nvvm':
        return get_cuda_paths()['nvvm'].by
    elif lib == 'libdevice':
        return get_cuda_paths()['libdevice'].by
    else:
        return get_cuda_paths()['cudalib_dir'].by


def test(_platform=None, print_paths=True):
    """Test library lookup.  Path info is printed to stdout.
    """
    failed = False
    libs = 'cublas cusparse cufft curand nvvm cudart'.split()
    for lib in libs:
        path = get_cudalib(lib, _platform)
        print('Finding {} from {}'.format(lib, _get_source_variable(lib)))
        if print_paths:
            print('\tlocated at', path)
        else:
            print('\tnamed ', os.path.basename(path))

        if _platform in (None, sys.platform):
            try:
                print('\ttrying to open library', end='...')
                open_cudalib(lib)
                print('\tok')
            except OSError as e:
                print('\tERROR: failed to open %s:\n%s' % (lib, e))
                failed = True

    archs = 'compute_20', 'compute_30', 'compute_35', 'compute_50'
    where = _get_source_variable('libdevice')
    print('Finding libdevice from', where)
    for arch in archs:
        print('\tsearching for', arch, end='...')
        path = get_libdevice(arch)
        if path:
            print('\tok')
        else:
            print('\tERROR: can\'t open libdevice for %s' % arch)
            failed = True
    return not failed
