from __future__ import print_function, division, absolute_import

from distutils.ccompiler import CCompiler, new_compiler
from distutils.command.build_ext import build_ext
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
        # Need to import it here since setuptools may monkeypatch it
        from distutils.dist import Distribution
        self._verbose = False
        self._compiler = new_compiler()
        customize_compiler(self._compiler)
        self._build_ext = build_ext(Distribution())
        self._build_ext.finalize_options()
        self._py_lib_dirs = self._build_ext.library_dirs
        self._py_include_dirs = self._build_ext.include_dirs
        self._math_info = np_misc.get_info('npymath')

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        # DEBUG will let Numpy spew many messages, so stick to INFO
        # to print commands executed by distutils
        log.set_threshold(log.INFO if value else log.WARN)

    def compile_objects(self, sources, output_dir,
                        include_dirs=(), depends=(), macros=(),
                        extra_cflags=None):
        """
        Compile the given source files into a separate object file each,
        all beneath the *output_dir*.  A list of paths to object files
        is returned.

        *macros* has the same format as in distutils: a list of 1- or 2-tuples.
        If a 1-tuple (name,), the given name is considered undefined by
        the C preprocessor.
        If a 2-tuple (name, value), the given name is expanded into the
        given value by the C preprocessor.
        """
        objects = self._compiler.compile(sources,
                                         output_dir=output_dir,
                                         include_dirs=include_dirs,
                                         depends=depends,
                                         macros=macros or [],
                                         extra_preargs=extra_cflags)
        return objects

    def link_shared(self, output, objects, libraries=(),
                    library_dirs=(), export_symbols=(),
                    extra_ldflags=None):
        """
        Create a shared library *output* linking the given *objects*
        and *libraries* (all strings).
        """
        output_dir, output_filename = os.path.split(output)
        self._compiler.link(CCompiler.SHARED_OBJECT, objects,
                            output_filename, output_dir,
                            libraries, library_dirs,
                            export_symbols=export_symbols,
                            extra_preargs=extra_ldflags)

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
        Get the include directories necessary to compile against the Python
        and Numpy C APIs.
        """
        return list(self._py_include_dirs) + self._math_info['include_dirs']

    def get_ext_filename(self, ext_name):
        """
        Given a C extension's module name, return its intended filename.
        """
        return self._build_ext.get_ext_filename(ext_name)


#
# Patch Numpy's exec_command() to avoid random crashes on Windows in test_pycc
# see https://github.com/numpy/numpy/pull/7614
# and https://github.com/numpy/numpy/pull/7862
#

def _patch_exec_command():
    # Patch the internal worker _exec_command()
    import numpy.distutils.exec_command as mod
    orig_exec_command = mod._exec_command
    mod._exec_command = _exec_command

def _exec_command(command, use_shell=None, use_tee = None, **env):
    """
    Internal workhorse for exec_command().
    Code from https://github.com/numpy/numpy/pull/7862
    """
    if use_shell is None:
        use_shell = os.name=='posix'
    if use_tee is None:
        use_tee = os.name=='posix'

    executable = None

    if os.name == 'posix' and use_shell:
        # On POSIX, subprocess always uses /bin/sh, override
        sh = os.environ.get('SHELL', '/bin/sh')
        if is_sequence(command):
            command = [sh, '-c', ' '.join(command)]
        else:
            command = [sh, '-c', command]
        use_shell = False

    elif os.name == 'nt' and _is_sequence(command):
        # On Windows, join the string for CreateProcess() ourselves as
        # subprocess does it a bit differently
        command = ' '.join(_quote_arg(arg) for arg in command)

    # Inherit environment by default
    env = env or None
    try:
        proc = subprocess.Popen(command, shell=use_shell, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True)
    except EnvironmentError:
        # Return 127, as os.spawn*() and /bin/sh do
        return '', 127
    text, err = proc.communicate()
    # Only append stderr if the command failed, as otherwise
    # the output may become garbled for parsing
    if proc.returncode:
        if text:
            text += "\n"
        text += err
    # Another historical oddity
    if text[-1:] == '\n':
        text = text[:-1]
    if use_tee:
        print(text)
    return proc.returncode, text


def _quote_arg(arg):
    """
    Quote the argument for safe use in a shell command line.
    """
    # If there is a quote in the string, assume relevants parts of the
    # string are already quoted (e.g. '-I"C:\\Program Files\\..."')
    if '"' not in arg and ' ' in arg:
        return '"%s"' % arg
    return arg


def _is_sequence(arg):
    if isinstance(arg, (str, bytes)):
        return False
    try:
        len(arg)
        return True
    except Exception:
        return False
