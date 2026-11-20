"""Discover runtimes that provide LLVM 22's vector-library ABI exports.

The export requirements must be reviewed when LLVM's mappings change.
"""

import ctypes
from ctypes.util import find_library
import functools
import platform
import sys

from llvmlite import binding as llvm


_LIBRARIES = {
    'libsystem': 'System',
    'accelerate': 'Accelerate',
    'svml': 'svml',
}


def _candidates():
    machine = platform.machine().lower()
    if sys.platform == 'darwin' and machine in ('arm64', 'x86_64'):
        return ('libsystem', 'accelerate') + (
            ('svml',) if machine == 'x86_64' else ())
    if (sys.platform.startswith(('linux', 'win')) and
            machine in ('x86_64', 'amd64')):
        return ('svml',)
    return ()


def _symbols(provider):
    if provider == 'svml':
        functions = 'sin cos tan pow exp exp2 log log2 log10 sqrt'.split()
        return tuple(f'__svml_{name}{suffix}' for name in functions
                     for suffix in ('2', '4', '8', 'f4', 'f8', 'f16'))
    if provider == 'accelerate':
        functions = ('ceil fabs floor sqrt exp expm1 log log1p log10 logb '
                     'sin cos tan asin acos atan atan2 sinh cosh tanh '
                     'asinh acosh atanh').split()
        return tuple(f'v{name}f' for name in functions)
    functions = ('exp acos asin atan atan2 cos sin tan cbrt erf pow '
                 'sinh cosh tanh asinh acosh atanh').split()
    return tuple(f'_simd_{name}_{suffix}' for name in functions
                 for suffix in ('d2', 'f4'))


@functools.lru_cache(maxsize=None)
def _load(provider):
    name = _LIBRARIES[provider]
    if provider == 'svml' and sys.platform == 'win32':
        name = 'svml_dispmd'
    path = find_library(name)
    if path is None:
        raise RuntimeError(f"Vector library {provider!r} could not be found")
    try:
        library = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
        for symbol in _symbols(provider):
            getattr(library, symbol)
        llvm.load_library_permanently(path)
    except (OSError, AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Cannot load vector library {provider!r} from {path!r}: {exc}"
        ) from exc
    return path


def resolve(requested='auto'):
    """Load a compatible runtime and return ``(provider, runtime_path)``."""
    choices = ('auto', 'none', *_LIBRARIES)
    if not isinstance(requested, str) or requested not in choices:
        raise ValueError(
            f"Invalid NUMBA_VECLIB value {requested!r}; "
            f"expected one of {choices}"
        )
    if requested == 'none':
        return ('none', None)
    candidates = _candidates()
    if requested != 'auto' and requested not in candidates:
        raise RuntimeError(
            f"Vector library {requested!r} is unsupported on "
            f"{sys.platform}/{platform.machine()}"
        )
    for provider in candidates if requested == 'auto' else (requested,):
        try:
            runtime = _load(provider)
        except RuntimeError:
            if requested == 'auto':
                continue
            raise
        return ('darwin_libsystem_m' if provider == 'libsystem' else provider,
                runtime)
    return ('none', None)
