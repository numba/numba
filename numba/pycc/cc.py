from __future__ import print_function, division, absolute_import

from collections import defaultdict
import logging
import os
import shutil
import sys
import tempfile

from numba import sigutils
from .compiler import ModuleCompiler, ExportEntry
from .platform import Toolchain


class CC(object):
    # NOTE: using ccache can speed up repetitive builds
    # (especially for the mixin modules)

    _mixin_sources = ['modulemixin.c', '../_math_c99.c']

    # -flto strips all unused helper functions, which 1) makes the
    # produced output much smaller and 2) can make the linking step faster.
    # (the Windows linker seems to do this by default, judging by the results)

    _extra_cflags = {
        'posix': ['-flto'],
        }

    _extra_ldflags = {
        'posix': ['-flto'],
        }

    def __init__(self, basename, source_module=None):
        if '.' in basename:
            raise ValueError("basename should be a simple module name, not "
                             "qualified name")

        self._basename = basename
        self._init_function = 'pycc_init_' + basename
        self._exported_functions = {}
        # Resolve source module name and directory
        f = sys._getframe(1)
        if source_module is None:
            dct = f.f_globals
            source_module = dct['__name__']
        elif hasattr(source_module, '__name__'):
            dct = source_module.__dict__
            source_module = source_module.__name__
        else:
            dct = sys.modules[source_module].__dict__
        source_dir = os.path.dirname(dct.get('__file__', ''))

        self._source_module = source_module
        self._toolchain = Toolchain()
        self._debug = False
        # By default, output in directory of caller module
        self._output_dir = source_dir
        self._output_file = self._toolchain.get_ext_filename(basename)
        self._use_nrt = False

    @property
    def name(self):
        return self._basename

    @property
    def output_file(self):
        return self._output_file

    @output_file.setter
    def output_file(self, value):
        self._output_file = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    @property
    def use_nrt(self):
        return self._use_nrt

    @use_nrt.setter
    def use_nrt(self, value):
        self._use_nrt = value

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value
        self._toolchain.debug = value

    def export(self, exported_name, sig):
        sig = sigutils.parse_signature(sig)
        if exported_name in self._exported_functions:
            raise KeyError("duplicated export symbol %s" % (exported_name))

        def decorator(func):
            entry = ExportEntry(exported_name, sig, func)
            self._exported_functions[exported_name] = entry
            return func

        return decorator

    @property
    def _export_entries(self):
        return sorted(self._exported_functions.values(),
                      key=lambda entry: entry.symbol)

    def _compile_mixins(self, build_dir):
        here = os.path.dirname(__file__)
        mixin_sources = self._mixin_sources[:]
        if self._use_nrt:
            mixin_sources.append('../runtime/nrt.c')
        sources = [os.path.join(here, f) for f in mixin_sources]
        include_dirs = self._toolchain.get_python_include_dirs()
        # Inject macro definitions required by modulemixin.c
        macros = [
            ('PYCC_MODULE_NAME', self._basename),
            ('PYCC_USE_NRT', int(self._use_nrt)),
            ]

        extra_cflags = self._extra_cflags.get(sys.platform, [])
        if not extra_cflags:
            extra_cflags = self._extra_cflags.get(os.name, [])
        # XXX distutils creates a whole subtree inside build_dir,
        # e.g. /tmp/test_pycc/home/antoine/numba/numba/pycc/modulemixin.o
        objects = self._toolchain.compile_objects(sources, build_dir,
                                                  include_dirs=include_dirs,
                                                  macros=macros,
                                                  extra_cflags=extra_cflags)
        return objects

    def compile(self):
        compiler = ModuleCompiler(self._export_entries, self._basename,
                                  self._use_nrt)
        compiler.external_init_function = self._init_function

        build_dir = tempfile.mkdtemp(prefix='pycc-build-%s-' % self._basename)

        # Compile object file
        temp_obj = os.path.join(build_dir,
                                os.path.splitext(self._output_file)[0] + '.o')
        compiler.write_native_object(temp_obj, wrap=True)
        objects = [temp_obj]

        # Compile mixins
        objects += self._compile_mixins(build_dir)

        # Then create shared library
        extra_ldflags = self._extra_ldflags.get(sys.platform, [])
        if not extra_ldflags:
            extra_ldflags = self._extra_ldflags.get(os.name, [])

        output_dll = os.path.join(self._output_dir, self._output_file)
        libraries = self._toolchain.get_python_libraries()
        library_dirs = self._toolchain.get_python_library_dirs()
        self._toolchain.link_shared(output_dll, objects,
                                    libraries, library_dirs,
                                    export_symbols=compiler.dll_exports,
                                    extra_ldflags=extra_ldflags)

        shutil.rmtree(build_dir)
