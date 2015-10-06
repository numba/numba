from __future__ import print_function, division, absolute_import

from distutils.ccompiler import CCompiler, new_compiler
from distutils.command.build_ext import build_ext
from distutils.dist import Distribution
from distutils.sysconfig import customize_compiler
from distutils import log

import numpy.distutils.misc_util as np_misc

import functools
import os
import subprocess
import sys


_configs = {
    # DLL suffix, Python C extension suffix
    'win': ('.dll', '.pyd'),
    'default': ('.so', '.so'),
}


def get_configs(arg):
    return _configs.get(sys.platform[:3], _configs['default'])[arg]


find_shared_ending = functools.partial(get_configs, 0)
find_pyext_ending = functools.partial(get_configs, 1)


class _DummyExtension(object):
    libraries = []


class Toolchain(object):

    def __init__(self):
        self.debug = False
        self._compiler = new_compiler()
        customize_compiler(self._compiler)
        self._build_ext = build_ext(Distribution())
        self._build_ext.finalize_options()
        self._py_lib_dirs = self._build_ext.library_dirs
        self._py_include_dirs = self._build_ext.include_dirs
        self._math_info = np_misc.get_info('npymath')

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value
        # DEBUG will let Numpy spew many messages, so stick to INFO
        # to print commands executed by distutils
        log.set_threshold(log.INFO if value else log.WARN)

    def compile_objects(self, sources, output_dir,
                        include_dirs=(), depends=(), macros=()):
        """
        """
        objects = self._compiler.compile(sources,
                                         output_dir=output_dir,
                                         include_dirs=include_dirs,
                                         depends=depends,
                                         macros=macros or [])
        return objects

    def link_shared(self, output, objects, libraries=(),
                    library_dirs=(), export_symbols=()):
        """
        Create a shared library *output* linking the given *objects*
        and *libraries* (all strings).
        """
        output_dir, output_filename = os.path.split(output)
        self._compiler.link(CCompiler.SHARED_OBJECT, objects,
                            output_filename, output_dir,
                            libraries, library_dirs,
                            export_symbols=export_symbols)

    def get_python_libraries(self):
        """
        Get the library arguments necessary to link with Python.
        """
        libs = self._build_ext.get_libraries(_DummyExtension())
        if sys.platform == 'win32':
            # Under Windows, need to link explicitly against the CRT,
            # as the MSVC compiler would implicitly do.
            # (XXX msvcrtd in pydebug mode?)
            libs = libs + ['msvcrt']
        return libs + self._math_info['libraries']

    def get_python_library_dirs(self):
        """
        Get the library directories necessary to link with Python.
        """
        return list(self._py_lib_dirs) + self._math_info['library_dirs']

    def get_python_include_dirs(self):
        """
        """
        return list(self._py_include_dirs) + self._math_info['include_dirs']

    def get_ext_filename(self, ext_name):
        """
        """
        return self._build_ext.get_ext_filename(ext_name)
