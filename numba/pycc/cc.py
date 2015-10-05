from __future__ import print_function, division, absolute_import

from collections import defaultdict
import logging
import os
import sys

from numba import sigutils
from .compiler import ModuleCompiler, ExportEntry
from .platform import Toolchain


class CC(object):

    def __init__(self, basename, source_module=None):
        if '.' in basename:
            raise ValueError("basename should be a simple module name, not "
                             "qualified name")

        self._basename = basename
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

    def compile(self):
        compiler = ModuleCompiler(self._export_entries, self._basename)
        # First compile object file
        temp_obj = os.path.join(self._output_dir,
                                os.path.splitext(self._output_file)[0] + '.o')
        output_obj = os.path.join(self._output_dir, self._output_file)
        compiler.write_native_object(temp_obj, wrap=True)

        # Then create shared library
        libraries = self._toolchain.get_python_libraries()
        library_dirs = self._toolchain.get_python_library_dirs()
        self._toolchain.link_shared(output_obj, [temp_obj],
                                    libraries, library_dirs,
                                    export_symbols=compiler.dll_exports)
        os.remove(temp_obj)
