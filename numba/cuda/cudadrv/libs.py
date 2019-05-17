"""Cudatoolkit libraries lookup utilities.

Cudatoolkit libraries can be available via the `cudatoolkit` conda package,
user supplied location from CUDA_HOME, or old deprecating NUMBAPRO_ prefixed
environment variables.
"""
from __future__ import print_function
import re
import os
import sys
import ctypes
import platform
from collections import namedtuple, defaultdict

from numba.findlib import find_lib, find_file
from .driver import get_numbapro_envvar

if sys.platform == 'win32':
    _dllopener = ctypes.WinDLL
elif sys.platform == 'darwin':
    _dllopener = ctypes.CDLL
else:
    _dllopener = ctypes.CDLL


def get_cuda_home(*subdirs):
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is not None:
        return os.path.join(cuda_home, *subdirs)


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


_env_path_tuple = namedtuple('_env_path_tuple', ['by', 'info'])


def get_conda_ctk():
    is_conda_env = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if not is_conda_env:
        return
    # Asssume the existence of NVVM to imply cudatoolkit installed
    paths = find_lib('nvvm')
    if not paths:
        return
    return os.path.join(sys.prefix, 'lib')


def _get_nvvm_path_decision():
    options = [
        ('NUMBAPRO_NVVM', get_numbapro_envvar('NUMBAPRO_NVVM')),
        ('NUMBAPRO_CUDALIB', get_numbapro_envvar('NUMBAPRO_CUDALIB')),
        ('Conda environment', get_conda_ctk()),
        ('CUDA_HOME', get_cuda_home('nvvm', 'lib')),
    ]
    by, libdir = _find_valid_path(options)
    return by, libdir


def _get_nvvm_path():
    by, libdir = _get_nvvm_path_decision()
    candidates = find_lib('nvvm', libdir)
    path = max(candidates) if candidates else None
    return _env_path_tuple(by, path)


def _find_valid_path(options):
    """Find valid path from *options*, which is a list of 2-tuple of
    (name, path).  Return first pair where *path* is not None; or, return
    ('Conda environment', None) for default conda path.
    """
    for by, data in options:
        if data is not None:
            return by, data
    else:
        raise RuntimeError("cuda libraries not found")


def _get_libdevice_path_decision():
    options = [
        ('NUMBAPRO_LIBDEVICE', get_numbapro_envvar('NUMBAPRO_LIBDEVICE')),
        ('NUMBAPRO_CUDALIB', get_numbapro_envvar('NUMBAPRO_CUDALIB')),
        ('Conda environment', get_conda_ctk()),
        ('CUDA_HOME', get_cuda_home('nvvm', 'libdevice')),
    ]
    by, libdir = _find_valid_path(options)
    return by, libdir


def _get_libdevice_paths():
    by, libdir = _get_libdevice_path_decision()
    # Search for pattern
    pat = r'libdevice(\.(?P<arch>compute_\d+))?(\.\d+)*\.bc$'
    candidates = find_file(re.compile(pat), libdir)
    # Grouping
    out = defaultdict(list)
    for path in candidates:
        m = re.search(pat, path)
        arch = m.group('arch')
        out[arch].append(path)
    # Keep only the max (most recent version) of the bitcode files.
    out = {k: max(v) for k, v in out.items()}
    return _env_path_tuple(by, out)


def _get_cudalib_dir_path_decision():
    options = [
        ('NUMBAPRO_CUDALIB', get_numbapro_envvar('NUMBAPRO_CUDALIB')),
        ('Conda environment', get_conda_ctk()),
        ('CUDA_HOME', get_cuda_home('lib')),
    ]
    by, libdir = _find_valid_path(options)
    return by, libdir


def _get_cudalib_dir():
    by, libdir = _get_cudalib_dir_path_decision()
    return _env_path_tuple(by, libdir)


def get_cuda_paths():
    """Returns a dictionary mapping component names to a 2-tuple
    of (source_variable, info).

    The returned dictionary will have the following keys and infos:
    - "nvvm": file_path
    - "libdevice": List[Tuple[arch, file_path]]
    - "cudalib_dir": directory_path

    Note: The result of the function is cached.
    """
    # Check cache
    if hasattr(get_cuda_paths, '_cached_result'):
        return get_cuda_paths._cached_result
    else:
        # Not in cache
        d = {
            'nvvm': _get_nvvm_path(),
            'libdevice': _get_libdevice_paths(),
            'cudalib_dir': _get_cudalib_dir(),
        }
        # Cache result
        get_cuda_paths._cached_result = d
        return d
