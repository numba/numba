import sys
import re
import os
from collections import defaultdict, namedtuple

from numba.config import IS_WIN32, IS_OSX
from numba.findlib import find_lib, find_file
from numba.cuda.envvars import get_numbapro_envvar


_env_path_tuple = namedtuple('_env_path_tuple', ['by', 'info'])


def _find_valid_path(options):
    """Find valid path from *options*, which is a list of 2-tuple of
    (name, path).  Return first pair where *path* is not None.
    If no valid path is found, return ('<unavailable>', None)
    """
    for by, data in options:
        if data is not None:
            return by, data
    else:
        return '<unavailable>', None


def _get_libdevice_path_decision():
    options = [
        ('NUMBAPRO_LIBDEVICE', get_numbapro_envvar('NUMBAPRO_LIBDEVICE')),
        ('NUMBAPRO_CUDALIB', get_numbapro_envvar('NUMBAPRO_CUDALIB')),
        ('Conda environment', get_conda_ctk()),
        ('CUDA_HOME', get_cuda_home('nvvm', 'libdevice')),
        ('System', get_system_ctk('nvvm', 'libdevice')),
    ]
    by, libdir = _find_valid_path(options)
    return by, libdir


def _nvvm_lib_dir():
    if IS_WIN32:
        return 'nvvm', 'bin'
    elif IS_OSX:
        return 'nvvm', 'lib'
    else:
        return 'nvvm', 'lib64'


def _get_nvvm_path_decision():
    options = [
        ('NUMBAPRO_NVVM', get_numbapro_envvar('NUMBAPRO_NVVM')),
        ('NUMBAPRO_CUDALIB', get_numbapro_envvar('NUMBAPRO_CUDALIB')),
        ('Conda environment', get_conda_ctk()),
        ('CUDA_HOME', get_cuda_home(*_nvvm_lib_dir())),
        ('System', get_system_ctk(*_nvvm_lib_dir())),
    ]
    by, path = _find_valid_path(options)
    return by, path


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


def _cudalib_path():
    if IS_WIN32:
        return 'bin'
    elif IS_OSX:
        return 'lib'
    else:
        return 'lib64'


def _get_cudalib_dir_path_decision():
    options = [
        ('NUMBAPRO_CUDALIB', get_numbapro_envvar('NUMBAPRO_CUDALIB')),
        ('Conda environment', get_conda_ctk()),
        ('CUDA_HOME', get_cuda_home(_cudalib_path())),
        ('System', get_system_ctk(_cudalib_path())),
    ]
    by, libdir = _find_valid_path(options)
    return by, libdir


def _get_cudalib_dir():
    by, libdir = _get_cudalib_dir_path_decision()
    return _env_path_tuple(by, libdir)


def get_system_ctk(*subdirs):
    """Return path to system-wide cudatoolkit; or, None if it doesn't exist.
    """
    # Linux?
    if sys.platform.startswith('linux'):
        # Is cuda alias to /usr/local/cuda?
        # We are intentionally not getting versioned cuda installation.
        base = '/usr/local/cuda'
        if os.path.exists(base):
            return os.path.join(base, *subdirs)


def get_conda_ctk():
    """Return path to directory containing the shared libraries of cudatoolkit.
    """
    is_conda_env = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if not is_conda_env:
        return
    # Asssume the existence of NVVM to imply cudatoolkit installed
    paths = find_lib('nvvm')
    if not paths:
        return
    # Use the directory name of the max path
    return os.path.dirname(max(paths))


def get_cuda_home(*subdirs):
    """Get paths of CUDA_HOME.
    If *subdirs* are the subdirectory name to be appended in the resulting
    path.
    """
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is not None:
        return os.path.join(cuda_home, *subdirs)


def _get_nvvm_path():
    by, path = _get_nvvm_path_decision()
    if by != 'NUMBAPRO_NVVM':
        candidates = find_lib('nvvm', path)
        path = max(candidates) if candidates else None
    return _env_path_tuple(by, path)


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
