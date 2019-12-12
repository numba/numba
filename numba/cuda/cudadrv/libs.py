"""Cudatoolkit libraries lookup utilities.

Cudatoolkit libraries can be available via the `cudatoolkit` conda package or a
user supplied location from CUDA_HOME or a system wide location.
"""
from __future__ import print_function
import os
import sys
import ctypes
import platform

from numba.findlib import find_lib
from numba.cuda.cuda_paths import get_cuda_paths

if sys.platform == 'win32':
    _dllopener = ctypes.WinDLL
elif sys.platform == 'darwin':
    _dllopener = ctypes.CDLL
else:
    _dllopener = ctypes.CDLL


def get_libdevice(arch):
    d = get_cuda_paths()
    paths = d['libdevice'].info
    return paths.get(arch, paths.get(None))


def open_libdevice(arch):
    with open(get_libdevice(arch), 'rb') as bcfile:
        return bcfile.read()


def get_cudalib(lib, platform=None):
    if lib == 'nvvm':
        return get_cuda_paths()['nvvm'].info
    else:
        libdir = get_cuda_paths()['cudalib_dir'].info

    candidates = find_lib(lib, libdir, platform)
    return max(candidates) if candidates else None


def open_cudalib(lib, ccc=False):
    path = get_cudalib(lib)
    if path is None:
        raise OSError('library %s not found' % lib)
    if ccc:
        return ctypes.CDLL(path)
    return _dllopener(path)


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
    libs = 'cublas cusparse cufft curand nvvm'.split()
    for lib in libs:
        path = get_cudalib(lib, _platform)
        print('Finding {} from {}'.format(lib, _get_source_variable(lib)))
        if path:
            if print_paths:
                print('\tlocated at', path)
            else:
                print('\tnamed ', os.path.basename(path))
        else:
            print('\tERROR: can\'t locate lib')
            failed = True

        if not failed and _platform in (None, sys.platform):
            try:
                print('\ttrying to open library', end='...')
                open_cudalib(lib, ccc=True)
                print('\tok')
            except OSError as e:
                print('\tERROR: failed to open %s:\n%s' % (lib, e))
                # NOTE: ignore failure of dlopen on cuBlas on OSX 10.5
                failed = True if not _if_osx_10_5() else False

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


def _if_osx_10_5():
    if sys.platform == 'darwin':
        vers = tuple(map(int, platform.mac_ver()[0].split('.')))
        if vers < (10, 6):
            return True
    return False
