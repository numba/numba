# -*- coding: utf-8 -*-

"""
Generate a package with IR implementations and tools.
"""

from __future__ import print_function, division, absolute_import

import os
from textwrap import dedent
from itertools import chain

from . import generator
from . import formatting
from . import astgen
from . import visitorgen
from . import naming

#------------------------------------------------------------------------
# Tools Flags
#------------------------------------------------------------------------

cython = 1

#------------------------------------------------------------------------
# Tools Resolution
#------------------------------------------------------------------------

class Tool(object):
    def __init__(self, codegens, flags=0, depends=[]):
        self.codegens = codegens
        self.flags = flags
        self.depends = depends

    def __repr__(self):
        return "Tool(codegens=[%s])" % ", ".join(map(str, self.codegens))

def resolve_tools(tool_list, mask, tools=None, seen=None):
    if tools is None:
        tools = []
        seen = set()

    for tool in tool_list:
        if not (tool.flags & mask) and tool not in seen:
            seen.add(tool)
            resolve_tools(tool.depends, mask, tools, seen)
            tools.append(tool)

    return tools

def enumerate_tools(feature_names, mask):
    tool_set = set(chain(*[features[name] for name in feature_names]))
    tools = resolve_tools(tool_set, mask)
    return tools

def enumerate_codegens(feature_names, mask):
    tools = enumerate_tools(feature_names, mask)
    codegens = list(chain(*[tool.codegens for tool in tools]))
    return codegens

#------------------------------------------------------------------------
# Tool Definitions
#------------------------------------------------------------------------

def make_codegen_dict(codegens):
    return dict((codegen.out_filename, codegen) for codegen in codegens)

all_codegens = astgen.codegens + visitorgen.codegens
gens = make_codegen_dict(all_codegens)

pxd_ast_tool        = Tool([gens[naming.nodes + ".pxd"]], flags=cython)
py_ast_tool         = Tool([gens[naming.nodes + ".py"]])

pxd_interface_tool  = Tool([gens[naming.interface + ".pxd"]], flags=cython,
                           depends=[pxd_ast_tool])
py_interface_tool   = Tool([gens[naming.interface + ".py"]],
                           depends=[py_ast_tool])

pxd_visitor_tool    = Tool([gens[naming.visitor + ".pxd"]], flags=cython,
                           depends=[pxd_interface_tool])
py_visitor_tool     = Tool([gens[naming.visitor + ".py"]],
                           depends=[py_interface_tool, pxd_visitor_tool])

pxd_transform_tool  = Tool([gens[naming.transformer + ".pxd"]], flags=cython,
                           depends=[pxd_interface_tool])
py_transformr_tool  = Tool([gens[naming.transformer + ".py"]],
                           depends=[py_interface_tool, pxd_transform_tool])

pxd_ast_tool.depends.extend([pxd_interface_tool, py_interface_tool])

#------------------------------------------------------------------------
# Feature Definitions & Entry Points
#------------------------------------------------------------------------

features = {
    'all': [py_ast_tool, py_visitor_tool, py_transformr_tool],
    'ast': [py_ast_tool],
    'visitor': [py_visitor_tool],
    'transformer': [py_transformr_tool],
}

def build_package(schema_filename, feature_names, output_dir, mask=0):
    """
    Build a package from the given schema and feature names in output_dir.

    :param mask: indicates which features to mask, e.g. specifying
                 'mask=build.cython' disables Cython support.
    """
    codegens = enumerate_codegens(feature_names, mask)
    disk_allocator = generator.generate_from_file(
        schema_filename, codegens, output_dir)

    try:
        _make_package(disk_allocator, codegens)
    finally:
        disk_allocator.close()

#------------------------------------------------------------------------
# Package Building Utilities
#------------------------------------------------------------------------

source_name = lambda fn: os.path.splitext(os.path.basename(fn))[0]

def _make_package(disk_allocator, codegens):
    _make_init(disk_allocator, codegens)

    # Make Cython dependency optional
    # disk_allocator.open_sourcefile("cython.py")

    fns = [c.out_filename for c in codegens if c.out_filename.endswith('.pxd')]
    if fns:
        _make_setup(disk_allocator, [source_name(fn) + '.py' for fn in fns])

def _make_init(disk_allocator, codegens):
    init = disk_allocator.open_sourcefile("__init__.py")
    init.write(dedent("""
        # Horrid hack to make work around circular cimports
        import os, sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    """))

    for c in codegens:
        if c.out_filename.endswith('.py'):
            modname = source_name(c.out_filename)
            init.write("from %s import *\n" % modname)

def _make_setup(disk_allocator, filenames):
    setup = disk_allocator.open_sourcefile("setup.py")

    ext_modules = ["Extension('%s', ['%s'])" % (source_name(fn), fn)
                      for fn in filenames]

    setup.write(dedent("""
        from distutils.core import setup
        from Cython.Distutils import build_ext
        from Cython.Distutils.extension import Extension

        ext_modules = [
            %s
        ]

        setup(
            # ext_modules=cythonize('*.pyx'),
            ext_modules=ext_modules,
            cmdclass={'build_ext': build_ext},
        )
    """) % formatting.py_formatter.format_stats(",\n", 4, ext_modules))
