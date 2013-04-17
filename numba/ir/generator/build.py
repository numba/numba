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

#------------------------------------------------------------------------
# Tool Definitions
#------------------------------------------------------------------------

def make_codegen_dict(codegens):
    return dict((codegen.out_filename, codegen) for codegen in codegens)

all_codegens = astgen.codegens + visitorgen.codegens
gens = make_codegen_dict(all_codegens)

pxd_ast_tool        = Tool([gens["nodes.pxd"]], flags=cython)
py_ast_tool         = Tool([gens["nodes.py"]])

pxd_interface_tool  = Tool([gens["interface.pxd"]], flags=cython,
                           depends=[pxd_ast_tool])
py_interface_tool   = Tool([gens["interface.py"]],
                           depends=[py_ast_tool])

pxd_visitor_tool    = Tool([gens["visitor.pxd"]], flags=cython,
                           depends=[pxd_interface_tool])
py_visitor_tool     = Tool([gens["visitor.py"]],
                           depends=[py_interface_tool, pxd_visitor_tool])

pxd_transform_tool  = Tool([gens["transformer.pxd"]], flags=cython,
                           depends=[pxd_interface_tool])
py_transformr_tool  = Tool([gens["transformer.py"]],
                           depends=[py_interface_tool, pxd_transform_tool])

pxd_ast_tool.depends.extend([pxd_interface_tool, py_interface_tool])

#------------------------------------------------------------------------
# Feature Definitions
#------------------------------------------------------------------------

features = {
    'all': [py_ast_tool, py_visitor_tool, py_transformr_tool],
    'ast': [py_ast_tool],
    'visitor': [py_visitor_tool],
    'transform': [py_transformr_tool],
}

def build_package(schema_filename, feature_names, output_dir, mask=0):
    tool_set = set(chain(*[features[name] for name in feature_names]))
    tools = resolve_tools(tool_set, mask)
    codegens = list(chain(*[tool.codegens for tool in tools]))
    disk_allocator = generator.generate_from_file(
        schema_filename, codegens, output_dir)

    try:
        make_package(disk_allocator, codegens)
    finally:
        disk_allocator.close()

#------------------------------------------------------------------------
# Package building
#------------------------------------------------------------------------

source_name = lambda fn: os.path.splitext(os.path.basename(fn))[0]

def make_package(disk_allocator, codegens):
    init = disk_allocator.open_sourcefile("__init__.py")
    # for c in codegens:
    #     if c.out_filename.endswith('.py'):
    #         modname = source_name(c.out_filename)
    #         init.write("from .%s import *\n" % modname)

    fns = [c.out_filename for c in codegens if c.out_filename.endswith('.pxd')]
    if fns:
        make_setup(disk_allocator, [source_name(fn) + '.py' for fn in fns])

def make_setup(disk_allocator, filenames):
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
    """) % formatting.format_stats(",\n", 4, ext_modules))

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    testdir = os.path.join(root, "tests")
    schema_filename = os.path.join(testdir, "testschema1.asdl")
    features_names = ['ast', 'visitor', 'transform']
    build_package(schema_filename, features_names, os.path.join(root, "out"))