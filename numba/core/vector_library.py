"""Discover runtimes that provide LLVM 22's vector-library ABI exports.

The export requirements must be reviewed when LLVM's mappings change.
"""

import ctypes
from ctypes.util import find_library
import functools
import platform
import sys

from llvmlite import binding as llvm


# Map vector library names to spellings used for library discovery.
_LIBRARIES = {
    'darwin_libsystem_m': 'System',
    'accelerate': 'Accelerate',
    'svml': 'svml',
}


# Map platform and architecture pairs to compatible vector math libraries. For
# 'auto', libraries are tried in order until one of them loads. Otherwise,
# 'none' is selected. Explicitly requested libraries must also be listed here
# for the current platform.
_PLATFORM_VECTOR_LIBRARIES = {
    ('darwin', 'arm64'): ('darwin_libsystem_m', 'accelerate'),
    # TODO: add support for other supported platform libraries and test.
    # ('linux', 'x86_64'): ('libmvec', 'svml', 'amdlibm'),
    # ('linux', 'amd64'): ('libmvec', 'svml', 'amdlibm'),
    # ('linux', 'aarch64'): ('libmvec', 'sleefgnuabi', 'armpl'),
    # ('linux', 'riscv64'): ('sleefgnuabi',),
    # ('linux', 'ppc64le'): ('massv',),
    # ('win32', 'x86_64'): ('svml',),
    # ('win32', 'amd64'): ('svml',),
}


def _candidates():
    return _PLATFORM_VECTOR_LIBRARIES.get(
        (sys.platform, platform.machine().lower()), ())


def _symbols(vector_library):
    """Return the expected LLVM vector ABI symbols for a vector_library."""
    if vector_library == 'svml':
        functions = 'sin cos tan pow exp exp2 log log2 log10 sqrt'.split()
        return tuple(f'__svml_{name}{suffix}' for name in functions
                     for suffix in ('2', '4', '8', 'f4', 'f8', 'f16'))
    if vector_library == 'accelerate':
        functions = ('ceil fabs floor sqrt exp expm1 log log1p log10 logb '
                     'sin cos tan asin acos atan atan2 sinh cosh tanh '
                     'asinh acosh atanh').split()
        return tuple(f'v{name}f' for name in functions)
    functions = ('exp acos asin atan atan2 cos sin tan cbrt erf pow '
                 'sinh cosh tanh asinh acosh atanh').split()
    return tuple(f'_simd_{name}_{suffix}' for name in functions
                 for suffix in ('d2', 'f4'))


@functools.lru_cache(maxsize=None)
def _load(vector_library):
    """Load the runtime shared library for LLVM JIT symbol resolution."""
    name = _LIBRARIES[vector_library]
    if vector_library == 'svml' and sys.platform == 'win32':
        name = 'svml_dispmd'
    path = find_library(name)
    if path is None:
        raise RuntimeError(
            f"Vector library {vector_library!r} could not be found"
        )
    try:
        library = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
        for symbol in _symbols(vector_library):
            getattr(library, symbol)
        llvm.load_library_permanently(path)
    except (OSError, AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Cannot load vector library {vector_library!r} "
            f"from {path!r}: {exc}"
        ) from exc
    return path


def resolve(requested='auto'):
    """Resolve a vector library and return its LLVM name and runtime path."""
    choices = ('auto', 'none', *_LIBRARIES)
    if not isinstance(requested, str) or requested not in choices:
        available = ('auto', 'none', *_candidates())
        raise ValueError(
            f"Invalid NUMBA_VECLIB value {requested!r}; "
            f"expected one of {available}"
        )
    if requested == 'none':
        return ('none', None)
    candidates = _candidates()
    if requested != 'auto' and requested not in candidates:
        raise RuntimeError(
            f"Vector library {requested!r} is unsupported on "
            f"{sys.platform}/{platform.machine()}"
        )
    for vector_library in candidates if requested == 'auto' else (requested,):
        try:
            runtime = _load(vector_library)
        except RuntimeError:
            if requested == 'auto':
                continue
            raise
        return vector_library, runtime
    return ('none', None)
